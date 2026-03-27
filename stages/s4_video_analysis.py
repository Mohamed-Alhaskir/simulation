"""
Stage 4: Video-Based Non-Verbal Behaviour Analysis (LUCAS-Aligned)
====================================================================
Uses MediaPipe Tasks API (FaceLandmarker + PoseLandmarker + HandLandmarker)
for tracking face, pose, and hands — compatible with mediapipe 0.10.14.

Only computes features relevant to LUCAS non-verbal assessment items:

  D) Non-verbal behaviour
     D1 — Eye-contact
     D2 — Positioning
     D3 — Posture
     D4 — Facial expressions
     D5 — Gestures & mannerisms

  I) Professional behaviour (demeanour cues observable from video)

All other LUCAS items (A, B, C, E, F, G, H, J) are verbal / audio domain
and are NOT assessed by this stage.

══════════════════════════════════════════════════════════════════════════
Principles:
══════════════════════════════════════════════════════════════════════════

1. PERSON-RELATIVE BASELINES — All metrics are normalised against each
   individual's own resting/baseline values computed from the first N
   seconds of video.

2. DISTRIBUTION-BASED REPORTING — Reports continuous distributions
   (mean, SD, percentiles) instead of binary classifications.

3. VALIDATED EXPRESSION FEATURES — Facial expressions use MediaPipe
   blendshape coefficients rather than ad-hoc landmark-distance heuristics.

4. TEMPORAL PATTERN ANALYSIS — Fidgeting detected via autocorrelation /
   periodicity analysis distinguishing repetitive nervous movements from
   natural movement variety.

5. RELIABILITY INDICATORS — Every metric includes a confidence /
   reliability score so downstream LLM scoring can weight unreliable
   measurements appropriately.

CHANGELOG
---------
- Two-stage face detection: OpenCV DNN face detector (res10 SSD) finds the
  face bounding box on the full frame first. The face crop is then padded,
  upscaled to face_crop_size px, and passed to MediaPipe FaceLandmarker.
  Face landmark coordinates are remapped from crop-normalised space back to
  full-frame normalised space so all downstream geometry remains correct.
- dnn_interval (default 5): run OpenCV DNN face detector every N frames.
- calibration_seconds (default 120): baseline computed from first N seconds.
- All dead / commented-out code removed. Only metrics that appear in the
  JSON output are computed.
- D1: replaced head-pose yaw/pitch gaze proxy with iris-landmark-based gaze
  direction. Iris offset relative to eye corners gives true gaze direction
  independent of head orientation. Face detection rate is now reported as
  data_availability_rate, NOT as an eye-contact proxy — face non-detection
  is ambiguous (patient head turn vs clinician looking away).
- D2: method_note updated to reflect egocentric (head-mounted) camera on a
  seated patient. Camera pitch is approximately stable; eye_level_y is a
  valid positioning proxy with deviation from session baseline reported.
- D2: ROLLING HORIZON — horizon is now estimated every HORIZON_ROLLING_INTERVAL_S
  seconds across the full session (skipping the first HORIZON_SKIP_S seconds for
  camera settling). Per-frame horizon values are produced by linear interpolation
  between valid estimates. Frames are flagged as horizon_unreliable when the local
  std of nearby estimates exceeds HORIZON_INSTABILITY_STD (patient head movement).
  Frames beyond HORIZON_INTERP_MAX_GAP_S from any valid estimate receive
  horizon=None and are excluded from at/above eye-level counts.
- D5: head movement periodicity removed. Camera is head-mounted on patient;
  head_yaw/pitch deltas reflect patient head movements (nodding) not
  clinician movement — the two are indistinguishable. Hand periodicity
  retained as fidgeting indicator.
- Item I: head_movement_periodicity replaced with hand_movement_periodicity.
"""

import json, math, os, urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from stages.base import BaseStage
from utils.json_utils import JSONEncoder


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ---------------------------------------------------------------------------
# MediaPipe model downloads
# ---------------------------------------------------------------------------
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
_MODELS = {
    "face": (
        f"{_MODEL_BASE}/face_landmarker/face_landmarker/float32/1/face_landmarker.task",
        "face_landmarker.task",
    ),
    "pose": (
        f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_heavy/float32/1/pose_landmarker_heavy.task",
        "pose_landmarker_heavy.task",
    ),
    "hand": (
        f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float32/1/hand_landmarker.task",
        "hand_landmarker.task",
    ),
}
_MODEL_DIR = Path.home() / ".cache" / "mediapipe" / "models"


def _ensure_model(key: str) -> str:
    url, filename = _MODELS[key]
    path = _MODEL_DIR / filename
    if path.exists():
        return str(path)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[VideoAnalysis] Downloading {filename} ...")
    urllib.request.urlretrieve(url, str(path))
    print(f"[VideoAnalysis] Model saved to {path}")
    return str(path)


# ---------------------------------------------------------------------------
# OpenCV DNN face detector model (res10 SSD)
# ---------------------------------------------------------------------------
_DNN_MODELS = {
    "prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
        "face_detector/deploy.prototxt",
        "deploy.prototxt",
    ),
    "caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector"
        "_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel",
    ),
}


def _ensure_dnn_model() -> Tuple[str, str]:
    paths = {}
    for key, (url, filename) in _DNN_MODELS.items():
        path = _MODEL_DIR / filename
        if not path.exists():
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[VideoAnalysis] Downloading DNN face model: {filename} ...")
            urllib.request.urlretrieve(url, str(path))
            print(f"[VideoAnalysis] DNN model saved to {path}")
        paths[key] = str(path)
    return paths["prototxt"], paths["caffemodel"]


# ---------------------------------------------------------------------------
# Centralised behavioural thresholds
# ---------------------------------------------------------------------------
IRIS_HORIZONTAL_THRESHOLD = 0.175
IRIS_VERTICAL_THRESHOLD   = 0.175

# Iris offset plausibility bounds — values outside these indicate a
# degenerate/occluded detection and are excluded from distributions.
# Horizontal is tightly bounded; vertical can legitimately exceed 1.0
# slightly when the clinician looks up, but extreme values (>1.5 / <-0.5)
# are measurement noise from near-profile face angles.
IRIS_H_VALID_MIN = -0.5
IRIS_H_VALID_MAX =  1.5
IRIS_V_VALID_MIN = -0.5
IRIS_V_VALID_MAX =  1.5

# Arm openness values above this threshold indicate a wrist landmark
# has been detected far off-body (misdetection artefact) — exclude from
# distribution statistics. The hard clamp at 5.0 is retained as a safety
# floor in _extract_positioning_posture but values > ARM_OPENNESS_MAX are
# filtered out before computing D3 summary stats.
ARM_OPENNESS_MAX = 3.0

ARM_CROSSED_DEV         = -1.0
ARM_CROSSED_DEV_WARN    = -0.5
ARM_CROSSED_ABS         = 0.5

MIN_DOMAIN_SD = {
    "smile": 0.005,
}

# ---------------------------------------------------------------------------
# Horizon line estimation constants (D2 positioning)
# ---------------------------------------------------------------------------
HOUGH_RHO             = 1
HOUGH_THETA           = np.pi / 180
HOUGH_THRESHOLD       = 80
HOUGH_MIN_LINE_LEN    = 80
HOUGH_MAX_LINE_GAP    = 20
HORIZON_MAX_ANGLE_DEG = 15.0
HORIZON_VALID_RANGE   = (0.05, 0.95)

# Skip the first N seconds — camera is still settling on the patient's head.
HORIZON_SKIP_S = 10

# Rolling horizon: sample one frame every N seconds throughout the session.
HORIZON_ROLLING_INTERVAL_S = 5.0

# When assessing local stability, look at estimates within this half-window.
# A std above HORIZON_INSTABILITY_STD in this neighbourhood means the patient
# was moving their head at that moment → flag the frame's horizon as unreliable.
HORIZON_LOCAL_WINDOW_S     = 15.0
HORIZON_INSTABILITY_STD    = 0.06

# Beyond this gap from any valid estimate, we have no horizon information
# and the frame's horizon is set to None (excluded from D2 counts).
HORIZON_INTERP_MAX_GAP_S   = 15.0

HORIZON_MIN_VALID_SAMPLES  = 5

HORIZON_PERSON_MASK_W = 0.50
HORIZON_PERSON_MASK_H = 0.60

HORIZON_AT_LEVEL_THRESHOLD = 0.04




# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _distribution_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None,
            "percentiles": {"5th": None, "25th": None, "75th": None, "95th": None},
        }
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None,
            "percentiles": {"5th": None, "25th": None, "75th": None, "95th": None},
            "filtered_count": len(values),
        }
    arr = np.array(clean, dtype=float)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "percentiles": {
            "5th":  round(float(np.percentile(arr, 5)),  3),
            "25th": round(float(np.percentile(arr, 25)), 3),
            "75th": round(float(np.percentile(arr, 75)), 3),
            "95th": round(float(np.percentile(arr, 95)), 3),
        },
    }


def _proportion_and_count(condition_list: List[bool]) -> Dict[str, Any]:
    if not condition_list:
        return {"rate": None, "count": 0, "total": 0}
    total = len(condition_list)
    count = sum(condition_list)
    return {
        "rate": round(count / total, 3),
        "count": count,
        "total": total,
    }


def _value_distribution(values: List[str]) -> Dict[str, float]:
    if not values:
        return {}
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = len(values)
    return {k: round(v / total, 3) for k, v in counts.items()}


def _reliability_level(detection_rate: float) -> str:
    if detection_rate >= 0.75:
        return "high"
    elif detection_rate >= 0.40:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Horizon line estimation from room geometry (D2 positioning)
# ---------------------------------------------------------------------------

def _estimate_horizon_y_from_frame(
    frame_bgr: np.ndarray,
    person_mask_w: float = HORIZON_PERSON_MASK_W,
    person_mask_h: float = HORIZON_PERSON_MASK_H,
) -> Optional[float]:
    """
    Estimate the horizon Y position (normalised 0-1) from a single video frame
    using Hough probabilistic line detection on background regions.

    Algorithm:
      1. Convert to greyscale and apply Canny edge detection.
      2. Mask out the central person region to avoid body edges confounding
         the structural line detection.
      3. Run probabilistic Hough transform to detect line segments.
      4. Filter to near-horizontal lines (angle < HORIZON_MAX_ANGLE_DEG).
      5. For each line, compute its Y value at horizontal frame centre (x=0.5).
      6. Return the weighted median Y, weighted by line length.

    Returns:
      Normalised Y of estimated horizon (0.0=top, 1.0=bottom), or None if
      insufficient lines found.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Mask out the central person region
    mask = np.ones_like(edges)
    cx1 = int(w * (0.5 - person_mask_w / 2))
    cx2 = int(w * (0.5 + person_mask_w / 2))
    cy1 = int(h * (0.5 - person_mask_h / 2))
    cy2 = int(h * (0.5 + person_mask_h / 2))
    mask[cy1:cy2, cx1:cx2] = 0
    edges = edges * mask

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None or len(lines) == 0:
        return None

    horizon_ys: List[float] = []
    weights: List[float]    = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        if angle_deg > HORIZON_MAX_ANGLE_DEG:
            continue

        slope       = dy / dx
        y_at_centre = y1 + slope * (w / 2 - x1)
        y_norm      = y_at_centre / h
        if not (HORIZON_VALID_RANGE[0] <= y_norm <= HORIZON_VALID_RANGE[1]):
            continue

        line_length = math.sqrt(dx * dx + dy * dy)
        horizon_ys.append(y_norm)
        weights.append(line_length)

    if len(horizon_ys) < 3:
        return None

    ys_arr  = np.array(horizon_ys)
    wts_arr = np.array(weights)
    sort_idx    = np.argsort(ys_arr)
    sorted_ys   = ys_arr[sort_idx]
    sorted_wts  = wts_arr[sort_idx]
    cumulative  = np.cumsum(sorted_wts)
    median_idx  = np.searchsorted(cumulative, cumulative[-1] / 2)
    horizon_y   = float(sorted_ys[min(median_idx, len(sorted_ys) - 1)])
    return round(horizon_y, 4)


# ---------------------------------------------------------------------------
# Rolling horizon: build a per-frame horizon lookup for the full session
# ---------------------------------------------------------------------------

def _build_rolling_horizon_lookup(
    video_path: str,
    video_fps: float,
    total_frames: int,
) -> Dict[str, Any]:
    """
    Estimate the horizon Y on a rolling basis throughout the entire session.

    Approach
    --------
    1. Skip the first HORIZON_SKIP_S seconds (camera settling on patient's head).
    2. Sample one frame every HORIZON_ROLLING_INTERVAL_S seconds for the rest
       of the video.
    3. Run Hough horizon detection on each sampled frame.
    4. Build a time-series of (timestamp_s, horizon_y) for all valid samples.
    5. For every frame in the video:
         a. Find the two nearest valid samples (one before, one after).
         b. If both exist and the gap between them is <= HORIZON_INTERP_MAX_GAP_S,
            linearly interpolate to get a per-frame horizon_y.
         c. Assess local stability: compute std of all valid sample estimates
            within HORIZON_LOCAL_WINDOW_S of this frame. If std exceeds
            HORIZON_INSTABILITY_STD, flag the frame as horizon_unreliable
            (patient head movement making the camera pitch unstable).
         d. If no valid sample exists within HORIZON_INTERP_MAX_GAP_S on either
            side, set horizon_y = None (excluded from D2 counts entirely).

    Returns
    -------
    Dict with:
      sample_timestamps   — timestamps of sampled frames (s)
      sample_horizons     — horizon_y at each sampled frame (None if detection failed)
      valid_sample_count  — number of samples with successful detection
      total_sample_count  — total frames sampled
      session_median_y    — robust median across all valid samples
      session_std_y       — std across all valid samples (head movement indicator)
      per_frame_lookup    — dict mapping frame_idx -> {
                              horizon_y: float | None,
                              horizon_reliable: bool,
                              interp_method: "interpolated" | "none"
                            }
      method_note         — human-readable explanation
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "valid": False,
            "per_frame_lookup": {},
            "method_note": "Could not open video for rolling horizon estimation.",
        }

    skip_frames  = int(HORIZON_SKIP_S * video_fps)
    start_frame  = min(skip_frames, total_frames - 1)
    interval_frames = max(1, int(HORIZON_ROLLING_INTERVAL_S * video_fps))

    sample_frame_indices = list(range(start_frame, total_frames, interval_frames))

    sample_timestamps: List[float]          = []
    sample_horizons: List[Optional[float]]  = []

    for frame_idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            sample_timestamps.append(frame_idx / video_fps)
            sample_horizons.append(None)
            continue
        h_y = _estimate_horizon_y_from_frame(frame)
        sample_timestamps.append(frame_idx / video_fps)
        sample_horizons.append(h_y)

    cap.release()

    valid_pairs = [
        (t, h) for t, h in zip(sample_timestamps, sample_horizons) if h is not None
    ]
    valid_sample_count = len(valid_pairs)
    total_sample_count = len(sample_timestamps)

    if valid_sample_count < HORIZON_MIN_VALID_SAMPLES:
        return {
            "valid": False,
            "sample_timestamps": sample_timestamps,
            "sample_horizons": sample_horizons,
            "valid_sample_count": valid_sample_count,
            "total_sample_count": total_sample_count,
            "per_frame_lookup": {},
            "method_note": (
                f"Rolling horizon estimation failed: only {valid_sample_count}/"
                f"{total_sample_count} samples were valid. Room may lack clear "
                f"horizontal lines. D2 positioning will fall back to session-median "
                f"baseline."
            ),
        }

    valid_ts  = np.array([t for t, _ in valid_pairs])
    valid_hs  = np.array([h for _, h in valid_pairs])
    session_median_y = float(np.median(valid_hs))
    session_std_y    = float(np.std(valid_hs))

    # ── Build per-frame lookup ──────────────────────────────────────────────
    # For each frame we need: interpolated horizon_y, reliability flag.
    # We iterate over all frames and use np.searchsorted on valid_ts for speed.
    per_frame_lookup: Dict[int, Dict[str, Any]] = {}
    half_window = HORIZON_LOCAL_WINDOW_S / 2.0

    for frame_idx in range(total_frames):
        t = frame_idx / video_fps

        # Find neighbouring valid samples
        pos = int(np.searchsorted(valid_ts, t))  # index where t would be inserted

        before_idx = pos - 1  # last valid sample at or before t
        after_idx  = pos      # first valid sample at or after t

        has_before = 0 <= before_idx < len(valid_ts)
        has_after  = 0 <= after_idx  < len(valid_ts)

        t_before = valid_ts[before_idx] if has_before else None
        t_after  = valid_ts[after_idx]  if has_after  else None
        h_before = valid_hs[before_idx] if has_before else None
        h_after  = valid_hs[after_idx]  if has_after  else None

        gap_before = (t - t_before) if has_before else float("inf")
        gap_after  = (t_after - t)  if has_after  else float("inf")
        min_gap    = min(gap_before, gap_after)

        if min_gap > HORIZON_INTERP_MAX_GAP_S:
            # Too far from any valid estimate — no usable horizon for this frame
            per_frame_lookup[frame_idx] = {
                "horizon_y":       None,
                "horizon_reliable": False,
                "interp_method":   "none",
            }
            continue

        # Linear interpolation between the two bracketing samples
        if has_before and has_after and gap_before <= HORIZON_INTERP_MAX_GAP_S and gap_after <= HORIZON_INTERP_MAX_GAP_S:
            span = t_after - t_before
            if span < 1e-6:
                horizon_y = float(h_before)
            else:
                alpha     = (t - t_before) / span
                horizon_y = float(h_before + alpha * (h_after - h_before))
            interp_method = "interpolated"
        elif has_before and gap_before <= HORIZON_INTERP_MAX_GAP_S:
            horizon_y     = float(h_before)
            interp_method = "nearest_before"
        else:
            horizon_y     = float(h_after)
            interp_method = "nearest_after"

        # Local stability: std of valid samples within ±half_window seconds
        local_mask = (valid_ts >= t - half_window) & (valid_ts <= t + half_window)
        local_hs   = valid_hs[local_mask]
        if len(local_hs) >= 2:
            local_std     = float(np.std(local_hs))
            is_reliable   = local_std <= HORIZON_INSTABILITY_STD
        else:
            # Only one nearby sample — can't assess stability, treat as reliable
            local_std   = 0.0
            is_reliable = True

        per_frame_lookup[frame_idx] = {
            "horizon_y":        round(horizon_y, 4),
            "horizon_reliable": is_reliable,
            "local_std":        round(local_std, 4),
            "interp_method":    interp_method,
        }

    # Summary stats on reliability
    reliable_count = sum(
        1 for v in per_frame_lookup.values()
        if v.get("horizon_reliable") and v.get("horizon_y") is not None
    )
    has_horizon_count = sum(
        1 for v in per_frame_lookup.values() if v.get("horizon_y") is not None
    )

    return {
        "valid": True,
        "sample_timestamps":   sample_timestamps,
        "sample_horizons":     sample_horizons,
        "valid_sample_count":  valid_sample_count,
        "total_sample_count":  total_sample_count,
        "session_median_y":    round(session_median_y, 4),
        "session_std_y":       round(session_std_y, 4),
        "frames_with_horizon": has_horizon_count,
        "frames_reliable":     reliable_count,
        "per_frame_lookup":    per_frame_lookup,
        "method_note": (
            f"Rolling horizon estimated every {HORIZON_ROLLING_INTERVAL_S}s across "
            f"full session (first {HORIZON_SKIP_S}s skipped for camera settling). "
            f"{valid_sample_count}/{total_sample_count} samples valid. "
            f"Per-frame horizon produced by linear interpolation between valid "
            f"neighbours; frames beyond {HORIZON_INTERP_MAX_GAP_S}s from any valid "
            f"sample receive horizon=None. Frames flagged unreliable where local "
            f"horizon std (within ±{HORIZON_LOCAL_WINDOW_S/2}s) exceeds "
            f"{HORIZON_INSTABILITY_STD} (patient head movement). "
            f"Session median horizon Y = {session_median_y:.3f} "
            f"(std={session_std_y:.3f})."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline computation
# ═══════════════════════════════════════════════════════════════════════════

class PersonBaseline:
    """
    Computes person-relative baselines from the first N seconds of video.
    Only sub-baselines with sufficient calibration frames are marked valid.
    """

    MIN_CALIBRATION_FRAMES = 5

    def __init__(self, calibration_frames: List[Dict[str, Any]]):
        self.valid = len(calibration_frames) > 0

        self.gaze_valid        = False
        self.baseline_yaw      = 0.0
        self.baseline_pitch    = 0.0
        self.baseline_yaw_std  = 1.0
        self.baseline_pitch_std = 1.0

        # ── Arm openness baseline ──
        arm_open = [
            f["positioning_and_posture"]["arm_openness"]
            for f in calibration_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("arm_openness") is not None
        ]
        self.arm_valid = len(arm_open) >= self.MIN_CALIBRATION_FRAMES
        self.baseline_arm_openness = float(np.mean(arm_open)) if self.arm_valid else 0.8

        # ── Smile baseline ──
        smiles = [
            f["facial_expression"]["blendshape_smile"]
            for f in calibration_frames
            if f.get("facial_expression") and f["facial_expression"].get("blendshape_smile") is not None
        ]
        self.expression_valid = len(smiles) >= self.MIN_CALIBRATION_FRAMES
        if self.expression_valid:
            self.baseline_smile = float(np.mean(smiles))
            smile_sd = float(np.std(smiles))
            if smile_sd < MIN_DOMAIN_SD["smile"]:
                self.expression_valid  = False
                self.baseline_smile_std = 1.0
            else:
                self.baseline_smile_std = smile_sd
        else:
            self.baseline_smile     = 0.0
            self.baseline_smile_std = 1.0

        # ── Eye level baseline (supplementary for horizon elevation) ──
        eye_ys = [
            f["positioning_and_posture"]["eye_level_y"]
            for f in calibration_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_y") is not None
        ]
        self.baseline_eye_y = float(np.median(eye_ys)) if eye_ys else None
        self.eye_y_valid    = len(eye_ys) >= self.MIN_CALIBRATION_FRAMES

        self.calibration_quality = {
            "total_calibration_frames": len(calibration_frames),
            "posture_arm_frames":        len(arm_open),
            "expression_frames":         len(smiles),
            "eye_level_y_frames":        len(eye_ys),
            "arm_baseline_valid":        self.arm_valid,
            "expression_baseline_valid": self.expression_valid,
            "eye_y_baseline_valid":      self.eye_y_valid,
            "minimum_required":          self.MIN_CALIBRATION_FRAMES,
        }

    def posture_deviation(self, arm_openness: float) -> Optional[Dict[str, Any]]:
        arm_dev = (arm_openness - self.baseline_arm_openness) if (self.arm_valid and arm_openness is not None) else None
        return {"arm_openness_deviation": round(arm_dev, 3) if arm_dev is not None else None}

    def expression_deviation(self, smile: float) -> Optional[Dict[str, float]]:
        if not self.expression_valid:
            return None
        smile_dev = smile - self.baseline_smile
        smile_z   = smile_dev / self.baseline_smile_std if self.baseline_smile_std > 0 else 0
        return {"smile_deviation": round(smile_dev, 3), "smile_z_score": round(smile_z, 2)}

    def eye_level_deviation(self, eye_level_y: float) -> Optional[Dict[str, Any]]:
        if not self.eye_y_valid or self.baseline_eye_y is None:
            return None
        deviation = eye_level_y - self.baseline_eye_y
        return {"eye_level_y_deviation": round(deviation, 3), "below_baseline": deviation > 0}

    def to_dict(self) -> Dict[str, Any]:
        if not self.valid:
            return {"valid": False, "calibration_quality": getattr(self, "calibration_quality", {})}
        return {
            "valid": True,
            "calibration_quality": self.calibration_quality,
            "posture": {
                "arm_valid":            self.arm_valid,
                "resting_arm_openness": round(self.baseline_arm_openness, 3) if self.arm_valid else None,
            },
            "expression": {
                "valid":         self.expression_valid,
                "resting_smile": round(self.baseline_smile, 4),
            },
            "positioning": {
                "eye_y_valid":                self.eye_y_valid,
                "session_median_eye_level_y": round(self.baseline_eye_y, 3) if self.baseline_eye_y is not None else None,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main stage class
# ═══════════════════════════════════════════════════════════════════════════

class VideoAnalysisStage(BaseStage):
    """LUCAS-aligned NVB video analysis stage."""

    # ── Face landmark indices ──
    LEFT_EYE_INNER  = 133;  RIGHT_EYE_INNER = 362
    LEFT_EYE_OUTER  = 33;   RIGHT_EYE_OUTER = 263

    # ── Iris landmark indices (MediaPipe 478-point model) ──
    LEFT_IRIS_CENTER  = 468
    LEFT_IRIS_RIGHT   = 469
    LEFT_IRIS_TOP     = 470
    LEFT_IRIS_LEFT    = 471
    LEFT_IRIS_BOTTOM  = 472
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_RIGHT  = 474
    RIGHT_IRIS_TOP    = 475
    RIGHT_IRIS_LEFT   = 476
    RIGHT_IRIS_BOTTOM = 477

    LEFT_EYE_TOP    = 159;  LEFT_EYE_BOTTOM  = 145
    RIGHT_EYE_TOP   = 386;  RIGHT_EYE_BOTTOM = 374

    # ── Pose landmark indices ──
    POSE_NOSE            = 0
    POSE_LEFT_SHOULDER   = 11;  POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_ELBOW      = 13;  POSE_RIGHT_ELBOW    = 14
    POSE_LEFT_WRIST      = 15;  POSE_RIGHT_WRIST    = 16
    POSE_LEFT_HIP        = 23;  POSE_RIGHT_HIP      = 24

    DEFAULT_CALIBRATION_SECONDS = 120

    # ═══════════════════════════════════════════════════════════════════════
    # run()
    # ═══════════════════════════════════════════════════════════════════════
    def run(self, ctx):
        cfg = self._get_stage_config("video_analysis")
        output_dir = Path(ctx["output_base"]) / "04_video_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not cfg.get("enabled", False):
            self.logger.info("Video analysis disabled in config, skipping.")
            return ctx

        self._validate_config(cfg)

        video_path = self._resolve_video_source(ctx, cfg)
        if video_path is None:
            self.logger.warning("No video source available, skipping video analysis.")
            return ctx

        self.logger.info(f"Video source: {video_path}")

        for key in ("face", "pose", "hand"):
            _ensure_model(key)
        _ensure_dnn_model()

        sample_fps = cfg.get("sample_fps", 10)

        results    = self._analyze_video(video_path, sample_fps, cfg)
        frame_data = results["frame_data"]
        video_fps  = results["metadata"]["video_fps"]
        total_frames = results["metadata"]["total_frames"]

        self.logger.info(
            f"Analyzed {results['metadata']['frames_analyzed']} frames "
            f"({results['metadata']['faces_detected']} with face detected)"
        )

        calibration_s      = cfg.get("calibration_seconds", self.DEFAULT_CALIBRATION_SECONDS)
        calibration_frames = [f for f in frame_data if f["timestamp_s"] <= calibration_s]
        baseline           = PersonBaseline(calibration_frames)
        self.logger.info(
            f"Baseline computed from {len(calibration_frames)} calibration "
            f"frames ({calibration_s}s window). Valid: {baseline.valid}"
        )

        if baseline.valid:
            for f in frame_data:
                self._enrich_frame_with_baseline(f, baseline)

        # ── D2: Build rolling horizon lookup ──────────────────────────────
        # Estimates horizon Y every HORIZON_ROLLING_INTERVAL_S seconds across
        # the full session. Per-frame values are produced by interpolation.
        # Frames with unstable local horizon (patient head movement) are
        # flagged as unreliable and excluded from D2 at/above eye-level counts.
        self.logger.info("[D2 Horizon] Building rolling horizon lookup ...")
        rolling_horizon = _build_rolling_horizon_lookup(
            video_path=video_path,
            video_fps=video_fps,
            total_frames=total_frames,
        )
        self.logger.info(
            f"[D2 Horizon] valid={rolling_horizon['valid']} "
            f"valid_samples={rolling_horizon.get('valid_sample_count')} "
            f"frames_with_horizon={rolling_horizon.get('frames_with_horizon')} "
            f"frames_reliable={rolling_horizon.get('frames_reliable')}"
        )

        nvb_metrics = self._compute_lucas_nvb_metrics(
            frame_data, baseline, video_fps, rolling_horizon
        )

        annotated_video_path = None
        if cfg.get("generate_annotated_video", True):
            annotated_video_path = str(output_dir / "annotated_video.mp4")
            self._generate_annotated_video(
                video_path=video_path,
                frame_data=frame_data,
                cached_landmarks=results.get("cached_landmarks", {}),
                output_path=annotated_video_path,
                cfg=cfg,
                rolling_horizon=rolling_horizon,
            )

        lucas_nvb_output = self._build_llm_output(
            nvb_metrics=nvb_metrics,
            baseline=baseline,
            metadata=results["metadata"],
        )

        features_path = output_dir / "video_features.json"
        with open(features_path, "w") as f:
            json.dump(lucas_nvb_output, f, indent=2, ensure_ascii=False, cls=JSONEncoder)

        ctx["artifacts"]["video_features"]      = lucas_nvb_output
        ctx["artifacts"]["video_features_path"] = str(features_path)
        if annotated_video_path:
            ctx["artifacts"]["annotated_video_path"] = annotated_video_path

        return ctx

    # ═══════════════════════════════════════════════════════════════════════
    # Video source resolution
    # ═══════════════════════════════════════════════════════════════════════
    def _resolve_video_source(self, ctx, cfg):
        inventory = ctx["artifacts"].get("inventory", {})
        quadrants = inventory.get("quadrants", {})

        # Priority: config setting > inventory setting > composite video
        # (Config allows user to override the default quadrant selection)

        preferred_quadrant = cfg.get("preferred_quadrant")
        if preferred_quadrant and preferred_quadrant in quadrants:
            self.logger.info(f"Video analysis quadrant from config: '{preferred_quadrant}'")
            return quadrants[preferred_quadrant]

        inventory_quadrant = inventory.get("video_analysis_quadrant")
        if inventory_quadrant:
            if inventory_quadrant in quadrants:
                self.logger.info(f"Video analysis quadrant from inventory: '{inventory_quadrant}'")
                return quadrants[inventory_quadrant]
            self.logger.warning(
                f"inventory.video_analysis_quadrant='{inventory_quadrant}' not found "
                f"in quadrants {list(quadrants.keys())}; falling back."
            )

        composite = ctx["artifacts"].get("composite_video")
        if composite and Path(composite).exists():
            self.logger.info("No quadrant available; using composite video.")
            return composite

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Config validation
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _validate_config(cfg: Dict) -> None:
        import warnings
        NUMERIC_RANGES = {
            "sample_fps":           (0.1, 30),
            "detection_confidence": (0.0, 1.0),
            "tracking_confidence":  (0.0, 1.0),
            "dnn_confidence":       (0.0, 1.0),
            "upscale_factor":       (1.0, 8.0),
            "face_crop_size":       (64, 1024),
            "face_pad":             (0.0, 2.0),
            "dnn_interval":         (1, 30),
            "calibration_seconds":  (10, 3600),
            "num_workers":          (1, 128),
        }
        for key, (lo, hi) in NUMERIC_RANGES.items():
            val = cfg.get(key)
            if val is not None and not (lo <= val <= hi):
                warnings.warn(
                    f"[VideoAnalysis] Config '{key}'={val} is outside "
                    f"expected range [{lo}, {hi}]. Results may be unreliable.",
                    UserWarning, stacklevel=3,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Parallel video analysis
    # ═══════════════════════════════════════════════════════════════════════
    def _analyze_video(self, video_path, sample_fps, cfg):
        import cv2
        from multiprocessing import get_context, cpu_count
        from functools import partial

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_interval  = 1
        available_cores = cpu_count() or 4
        num_workers     = cfg.get("num_workers", min(available_cores, 40))
        segment_size    = max(1, total_frames // num_workers)

        self.logger.info(
            f"[Parallel] {video_fps:.1f} FPS | {total_frames} frames | "
            f"interval={frame_interval} | workers={num_workers}"
        )

        proto, model = _ensure_dnn_model()
        model_paths  = {
            "face":      _ensure_model("face"),
            "pose":      _ensure_model("pose"),
            "hand":      _ensure_model("hand"),
            "dnn_proto": proto,
            "dnn_model": model,
        }
        segments = []
        for i in range(num_workers):
            start = i * segment_size
            end   = (i + 1) * segment_size if i < num_workers - 1 else total_frames
            segments.append((video_path, start, end, frame_interval, cfg, model_paths))

        ctx_mp = get_context("spawn")
        with ctx_mp.Pool(processes=num_workers) as pool:
            results = pool.map(partial(self._process_video_segment), segments)

        frame_data, cached_landmarks = [], {}
        faces_detected, frames_analyzed = 0, 0
        for r in sorted(results, key=lambda x: x["segment_start"]):
            frame_data.extend(r["frame_data"])
            faces_detected  += r["faces_detected"]
            frames_analyzed += r["frames_analyzed"]
            cached_landmarks.update(r.get("cached_landmarks", {}))

        frame_data.sort(key=lambda x: x["frame_idx"])

        return {
            "metadata": {
                "video_path":        video_path,
                "video_fps":         video_fps,
                "total_frames":      total_frames,
                "sample_fps":        sample_fps,
                "frames_analyzed":   frames_analyzed,
                "faces_detected":    faces_detected,
                "face_detection_rate": round(faces_detected / frames_analyzed, 3)
                                       if frames_analyzed > 0 else 0,
                "model":   "mediapipe_tasks_face+pose+hand_parallel",
                "workers": num_workers,
            },
            "frame_data":       frame_data,
            "cached_landmarks": cached_landmarks,
        }

    def _process_video_segment(self, args):
        import os, cv2, mediapipe as mp

        os.environ["OMP_NUM_THREADS"]         = "1"
        os.environ["TF_NUM_INTEROP_THREADS"]  = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"]  = "1"

        video_path, start_frame, end_frame, frame_interval, cfg, model_paths = args

        face_model_path = model_paths["face"]
        pose_model_path = model_paths["pose"]
        hand_model_path = model_paths["hand"]
        dnn_proto       = model_paths["dnn_proto"]
        dnn_model_file  = model_paths["dnn_model"]

        BaseOptions        = mp.tasks.BaseOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode
        FaceLandmarker     = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        PoseLandmarker     = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker     = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        det_conf       = cfg.get("detection_confidence", 0.4)
        trk_conf       = cfg.get("tracking_confidence",  0.4)
        dnn_conf       = cfg.get("dnn_confidence",        0.3)
        face_crop_size = cfg.get("face_crop_size",        320)
        face_pad       = cfg.get("face_pad",              0.4)
        upscale_factor = cfg.get("upscale_factor",        1.0)
        do_upscale     = upscale_factor > 1.0
        dnn_interval   = cfg.get("dnn_interval",            1)

        dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model_file)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=trk_conf,
            min_tracking_confidence=trk_conf,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=trk_conf,
            min_tracking_confidence=trk_conf,
        )
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=trk_conf,
            min_tracking_confidence=trk_conf,
        )

        cap       = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_idx                     = start_frame
        frames_analyzed, faces_detected = 0, 0
        frame_data, cached_landmarks  = [], {}
        _last_dnn_bbox                = None
        _last_dnn_conf                = 0.0

        face_lm = FaceLandmarker.create_from_options(face_options)
        pose_lm = PoseLandmarker.create_from_options(pose_options)
        hand_lm = HandLandmarker.create_from_options(hand_options)

        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    try:
                        timestamp_ms = int((frame_idx / video_fps) * 1000)
                        timestamp_s  = frame_idx / video_fps
                        orig_h, orig_w = frame.shape[:2]
                        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        if frame_idx % dnn_interval == 0:
                            blob = cv2.dnn.blobFromImage(
                                frame, scalefactor=1.0, size=(300, 300),
                                mean=(104.0, 177.0, 123.0), swapRB=False, crop=False,
                            )
                            dnn_net.setInput(blob)
                            detections    = dnn_net.forward()
                            best_bbox_norm = None
                            best_dnn_conf  = 0.0
                            for d in range(detections.shape[2]):
                                confidence = float(detections[0, 0, d, 2])
                                if confidence > dnn_conf and confidence > best_dnn_conf:
                                    best_dnn_conf  = confidence
                                    best_bbox_norm = detections[0, 0, d, 3:7]
                            if best_bbox_norm is not None:
                                _last_dnn_bbox = best_bbox_norm
                                _last_dnn_conf = best_dnn_conf
                        else:
                            best_bbox_norm = _last_dnn_bbox
                            best_dnn_conf  = _last_dnn_conf

                        face_result = None
                        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, orig_w, orig_h
                        used_crop = False

                        if best_bbox_norm is not None:
                            bx1 = int(best_bbox_norm[0] * orig_w)
                            by1 = int(best_bbox_norm[1] * orig_h)
                            bx2 = int(best_bbox_norm[2] * orig_w)
                            by2 = int(best_bbox_norm[3] * orig_h)
                            bw  = max(bx2 - bx1, 1)
                            bh  = max(by2 - by1, 1)
                            pad_x = int(bw * face_pad)
                            pad_y = int(bh * face_pad)
                            cx1 = max(0, bx1 - pad_x);  cy1 = max(0, by1 - pad_y)
                            cx2 = min(orig_w, bx2 + pad_x); cy2 = min(orig_h, by2 + pad_y)
                            crop_x1, crop_y1 = cx1, cy1
                            crop_x2, crop_y2 = cx2, cy2
                            crop_w = cx2 - cx1;  crop_h = cy2 - cy1
                            if crop_w > 10 and crop_h > 10:
                                face_crop = rgb_full[cy1:cy2, cx1:cx2]
                                face_crop_resized = cv2.resize(
                                    face_crop, (face_crop_size, face_crop_size),
                                    interpolation=cv2.INTER_CUBIC,
                                )
                                mp_face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_resized)
                                face_result   = face_lm.detect(mp_face_image)
                                used_crop     = True

                        if face_result is None or not face_result.face_landmarks:
                            full_for_face = rgb_full
                            if do_upscale:
                                full_for_face = cv2.resize(
                                    rgb_full,
                                    (int(orig_w * upscale_factor), int(orig_h * upscale_factor)),
                                    interpolation=cv2.INTER_CUBIC,
                                )
                            mp_full_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=full_for_face)
                            face_result   = face_lm.detect(mp_full_image)
                            crop_x1, crop_y1 = 0, 0
                            crop_x2, crop_y2 = orig_w, orig_h
                            used_crop = False

                        face_confirmed = bool(face_result and face_result.face_landmarks)
                        if not face_confirmed:
                            _last_dnn_bbox = None
                            _last_dnn_conf = 0.0
                            cached_landmarks[frame_idx] = {
                                "face_landmarks": None, "pose_landmarks": None,
                                "left_hand_landmarks": None, "right_hand_landmarks": None,
                                "face_crop": None,
                            }
                            frame_data.append({
                                "timestamp_s": round(timestamp_s, 2),
                                "frame_idx":   frame_idx,
                                "dnn_face_confidence": round(best_dnn_conf, 3),
                                "used_face_crop": used_crop,
                                "face_detected": False,
                                "pose_detected": False,
                                "eye_contact": None,
                                "positioning_and_posture": None,
                            })
                            frames_analyzed += 1
                            frame_idx += 1
                            continue

                        if do_upscale:
                            full_rgb_pose = cv2.resize(
                                rgb_full,
                                (int(orig_w * upscale_factor), int(orig_h * upscale_factor)),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            pose_img_w = int(orig_w * upscale_factor)
                            pose_img_h = int(orig_h * upscale_factor)
                        else:
                            full_rgb_pose = rgb_full
                            pose_img_w    = orig_w
                            pose_img_h    = orig_h

                        mp_pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=full_rgb_pose)
                        pose_result   = pose_lm.detect_for_video(mp_pose_image, timestamp_ms)
                        hand_result   = hand_lm.detect_for_video(mp_pose_image, timestamp_ms)

                        raw_face_landmarks = face_result.face_landmarks[0]
                        face_landmarks = None
                        if raw_face_landmarks:
                            if used_crop:
                                face_landmarks = self._remap_face_landmarks(
                                    raw_face_landmarks,
                                    crop_x1, crop_y1, crop_x2, crop_y2,
                                    orig_w, orig_h,
                                )
                            else:
                                face_landmarks = raw_face_landmarks

                        blendshapes = None
                        if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                            blendshapes = {
                                bs.category_name: round(bs.score, 4)
                                for bs in face_result.face_blendshapes[0]
                            }

                        pose_landmarks = (
                            pose_result.pose_landmarks[0]
                            if pose_result.pose_landmarks else None
                        )
                        left_hand_lms, right_hand_lms = None, None
                        if hand_result.hand_landmarks and hand_result.handedness:
                            for i, handedness_list in enumerate(hand_result.handedness):
                                label = handedness_list[0].category_name.lower()
                                if label == "left" and left_hand_lms is None:
                                    left_hand_lms = hand_result.hand_landmarks[i]
                                elif label == "right" and right_hand_lms is None:
                                    right_hand_lms = hand_result.hand_landmarks[i]

                        cached_landmarks[frame_idx] = {
                            "face_landmarks":       face_landmarks,
                            "pose_landmarks":       pose_landmarks,
                            "left_hand_landmarks":  left_hand_lms,
                            "right_hand_landmarks": right_hand_lms,
                            "face_crop": (crop_x1, crop_y1, crop_x2, crop_y2) if used_crop else None,
                        }

                        frame_features = {
                            "timestamp_s":        round(timestamp_s, 2),
                            "frame_idx":          frame_idx,
                            "dnn_face_confidence": round(best_dnn_conf, 3),
                            "used_face_crop":     used_crop,
                        }

                        faces_detected += 1
                        frame_features["face_detected"]    = True
                        frame_features["eye_contact"]      = self._extract_iris_gaze(face_landmarks, orig_w, orig_h)
                        if pose_landmarks and len(pose_landmarks) > 0:
                            frame_features["pose_detected"]           = True
                            frame_features["positioning_and_posture"] = self._extract_positioning_posture(
                                pose_landmarks, face_landmarks=face_landmarks,
                                img_w=pose_img_w, img_h=pose_img_h,
                            )
                        else:
                            frame_features["pose_detected"]           = False
                            frame_features["positioning_and_posture"] = None

                        frame_data.append(frame_features)
                        frames_analyzed += 1

                    except Exception as _frame_err:
                        frame_data.append({
                            "timestamp_s": round(frame_idx / video_fps, 2),
                            "frame_idx":   frame_idx,
                            "face_detected": False,
                            "pose_detected": False,
                            "eye_contact": None,
                            "positioning_and_posture": None,
                            "error": str(_frame_err),
                        })
                        frames_analyzed += 1

                frame_idx += 1

        finally:
            face_lm.close()
            pose_lm.close()
            hand_lm.close()
            cap.release()

        return {
            "segment_start":    start_frame,
            "frames_analyzed":  frames_analyzed,
            "faces_detected":   faces_detected,
            "frame_data":       frame_data,
            "cached_landmarks": cached_landmarks,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Face landmark coordinate remapping (crop → full frame)
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _remap_face_landmarks(landmarks, crop_x1, crop_y1, crop_x2, crop_y2, orig_w, orig_h):
        from types import SimpleNamespace
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w <= 0 or crop_h <= 0:
            return landmarks
        return [
            SimpleNamespace(
                x=(lm.x * crop_w + crop_x1) / orig_w,
                y=(lm.y * crop_h + crop_y1) / orig_h,
                z=getattr(lm, "z", 0.0),
                visibility=getattr(lm, "visibility", 1.0),
            )
            for lm in landmarks
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline enrichment
    # ═══════════════════════════════════════════════════════════════════════
    def _enrich_frame_with_baseline(self, frame: Dict, baseline: PersonBaseline):
        pp = frame.get("positioning_and_posture")
        if pp:
            arm   = pp.get("arm_openness")
            pp["baseline_deviation"] = baseline.posture_deviation(arm)
            eye_y = pp.get("eye_level_y")
            if eye_y is not None:
                pp["eye_level_baseline_deviation"] = baseline.eye_level_deviation(eye_y)


    # ═══════════════════════════════════════════════════════════════════════
    # D1 — Eye-contact via iris landmark gaze direction
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_iris_gaze(self, landmarks, img_w: int, img_h: int) -> Optional[Dict[str, Any]]:
        if landmarks is None or len(landmarks) < 478:
            return None

        def lm(idx):
            l = landmarks[idx]
            return np.array([l.x, l.y])

        def iris_offset(iris_center, inner, outer, top, bottom):
            eye_w = np.linalg.norm(outer - inner)
            eye_h = np.linalg.norm(bottom - top)
            if eye_w < 1e-4 or eye_h < 1e-4:
                return None, None
            h_axis  = (outer - inner) / eye_w
            v_axis  = (bottom - top)  / eye_h
            h_off   = float(np.dot(iris_center - inner, h_axis) / eye_w)
            v_off   = float(np.dot(iris_center - top,   v_axis) / eye_h)
            return h_off, v_off

        lh, lv = iris_offset(
            lm(self.LEFT_IRIS_CENTER),
            lm(self.LEFT_EYE_INNER), lm(self.LEFT_EYE_OUTER),
            lm(self.LEFT_EYE_TOP),   lm(self.LEFT_EYE_BOTTOM),
        )
        rh, rv = iris_offset(
            lm(self.RIGHT_IRIS_CENTER),
            lm(self.RIGHT_EYE_INNER), lm(self.RIGHT_EYE_OUTER),
            lm(self.RIGHT_EYE_TOP),   lm(self.RIGHT_EYE_BOTTOM),
        )

        # Plausibility filter: reject offsets from degenerate detections
        # (near-profile face, heavy occlusion) before averaging.
        def plausible_h(v): return v is not None and IRIS_H_VALID_MIN <= v <= IRIS_H_VALID_MAX
        def plausible_v(v): return v is not None and IRIS_V_VALID_MIN <= v <= IRIS_V_VALID_MAX

        h_vals = [o for o in [lh, rh] if plausible_h(o)]
        v_vals = [o for o in [lv, rv] if plausible_v(o)]
        if not h_vals:
            return None

        avg_h = float(np.mean(h_vals))
        avg_v = float(np.mean(v_vals)) if v_vals else None

        h_on  = abs(avg_h - 0.5) < IRIS_HORIZONTAL_THRESHOLD
        v_on  = (abs(avg_v - 0.5) < IRIS_VERTICAL_THRESHOLD) if avg_v is not None else True

        return {
            "iris_horizontal_offset": round(avg_h, 3),
            "iris_vertical_offset":   round(avg_v, 3) if avg_v is not None else None,
            "on_target":              h_on and v_on,
            "left_iris_h_offset":     round(lh, 3) if plausible_h(lh) else None,
            "right_iris_h_offset":    round(rh, 3) if plausible_h(rh) else None,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # D2 — Positioning  &  D3 — Posture
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_positioning_posture(self, pose_landmarks, face_landmarks=None, img_w=None, img_h=None) -> Dict[str, Any]:
        def plm(idx):
            lm = pose_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z]), getattr(lm, "visibility", 1.0)

        left_shoulder,  ls_vis = plm(self.POSE_LEFT_SHOULDER)
        right_shoulder, rs_vis = plm(self.POSE_RIGHT_SHOULDER)
        left_hip,       lh_vis = plm(self.POSE_LEFT_HIP)
        right_hip,      rh_vis = plm(self.POSE_RIGHT_HIP)
        left_wrist,     _      = plm(self.POSE_LEFT_WRIST)
        right_wrist,    _      = plm(self.POSE_RIGHT_WRIST)

        shoulder_width       = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_width_valid = shoulder_width >= 0.05
        wrist_distance       = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
        arm_openness         = min(wrist_distance / shoulder_width, 5.0) if shoulder_width_valid else None
        avg_vis              = float(np.mean([ls_vis, rs_vis, lh_vis, rh_vis]))

        # ── D2: Eye-level Y from face landmarks (used for horizon elevation) ──
        eye_level_y = None
        if face_landmarks and len(face_landmarks) > max(self.LEFT_EYE_INNER, self.RIGHT_EYE_INNER):
            eye_level_y = round(float(
                (face_landmarks[self.LEFT_EYE_INNER].y + face_landmarks[self.RIGHT_EYE_INNER].y) / 2
            ), 3)

        return {
            "arm_openness":         round(float(arm_openness), 3) if arm_openness is not None else None,
            "shoulder_width_valid": shoulder_width_valid,
            "eye_level_y":          eye_level_y,
            "landmark_confidence":  round(avg_vis, 3),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate LUCAS NVB metrics
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_lucas_nvb_metrics(
        self,
        frame_data: List[Dict],
        baseline: PersonBaseline,
        video_fps: float,
        rolling_horizon: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not frame_data:
            return {}

        total       = len(frame_data)
        face_frames = [f for f in frame_data if f.get("face_detected")]
        pose_frames = [f for f in frame_data if f.get("pose_detected")]
        face_rate   = len(face_frames) / total if total > 0 else 0
        pose_rate   = len(pose_frames) / total if total > 0 else 0

        # ── D1: Eye-contact ──────────────────────────────────────────────────
        gaze_frames          = [f for f in face_frames if f.get("eye_contact") and f["eye_contact"].get("on_target") is not None]
        gaze_on_target_flags = [f["eye_contact"]["on_target"] for f in gaze_frames]
        iris_h_offsets       = [f["eye_contact"]["iris_horizontal_offset"] for f in gaze_frames if f["eye_contact"].get("iris_horizontal_offset") is not None]
        iris_v_offsets       = [f["eye_contact"]["iris_vertical_offset"]   for f in gaze_frames if f["eye_contact"].get("iris_vertical_offset")   is not None]

        d1 = {
            "gaze_on_target":                    _proportion_and_count(gaze_on_target_flags),
            "iris_horizontal_offset_distribution": _distribution_summary(iris_h_offsets),
            "iris_vertical_offset_distribution":   _distribution_summary(iris_v_offsets),
            "data_availability_rate":              round(face_rate, 3),
            "gaze_measurable_frames":              len(gaze_frames),
            "total_frames":                        total,
            "reliability":                         _reliability_level(face_rate),
            "method_note": (
                "Gaze measured via iris landmark offset relative to eye corners "
                "(MediaPipe 478-point model). Offset 0.5 = iris centered = "
                "looking directly at camera (= at the patient). "
                "Thresholds: horizontal |offset - 0.5| < 0.175, vertical < 0.175. "
                "data_availability_rate = fraction of frames where face was detectable. "
                "Face non-detection is MISSING DATA — it may mean the patient turned "
                "their head, not that the clinician looked away. "
                "Only gaze_measurable_frames contribute to gaze_on_target."
            ),
        }

        # ── D2: Positioning — rolling Hough horizon (primary) ────────────────
        #
        # The horizon Y (normalised 0=top, 1=bottom) is estimated from room
        # geometry using Hough line detection on background regions, sampled
        # every HORIZON_ROLLING_INTERVAL_S seconds across the full session.
        # Per-frame values are produced by linear interpolation; frames where
        # the local horizon std exceeds HORIZON_INSTABILITY_STD (patient head
        # movement) are excluded as unreliable.
        #
        # elevation = horizon_y - eye_level_y
        #   > HORIZON_AT_LEVEL_THRESHOLD  → clinician above patient eye level (unfavourable)
        #   within ±HORIZON_AT_LEVEL_THRESHOLD → at eye level (favourable)

        horizon_lookup = rolling_horizon.get("per_frame_lookup", {})
        horizon_valid  = rolling_horizon.get("valid", False)

        eye_level_ys        = []
        horizon_elevations  = []
        at_eye_level_flags  = []
        above_horizon_flags = []

        for f in pose_frames:
            pp = f.get("positioning_and_posture")
            if not pp or pp.get("eye_level_y") is None:
                continue
            eye_y     = pp["eye_level_y"]
            frame_idx = f["frame_idx"]
            eye_level_ys.append(eye_y)

            if not horizon_valid:
                continue

            h_info     = horizon_lookup.get(frame_idx, {})
            h_y        = h_info.get("horizon_y")
            h_reliable = h_info.get("horizon_reliable", False)

            if h_y is None or not h_reliable:
                continue

            elevation = h_y - eye_y
            horizon_elevations.append(elevation)
            at_eye_level_flags.append(abs(elevation) < HORIZON_AT_LEVEL_THRESHOLD)
            above_horizon_flags.append(elevation > HORIZON_AT_LEVEL_THRESHOLD)

        # Eye-Y deviation from session baseline (supplementary)
        eye_level_deviations = [
            f["positioning_and_posture"]["eye_level_baseline_deviation"]["eye_level_y_deviation"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_baseline_deviation")
            and f["positioning_and_posture"]["eye_level_baseline_deviation"].get("eye_level_y_deviation") is not None
        ]

        # D2 reliability: based on horizon frame coverage
        horizon_coverage = len(horizon_elevations) / max(len(pose_frames), 1) if pose_frames else 0
        d2_reliability = _reliability_level(horizon_coverage)

        d2 = {
            "primary_method": "rolling_hough_horizon",
            "horizon_valid":                        horizon_valid,
            "session_median_horizon_y":             rolling_horizon.get("session_median_y"),
            "session_horizon_std":                  rolling_horizon.get("session_std_y"),
            "horizon_elevation_distribution":       _distribution_summary(horizon_elevations),
            "at_patient_eye_level_rate":            _proportion_and_count(at_eye_level_flags),
            "above_patient_eye_level_rate":         _proportion_and_count(above_horizon_flags),
            "frames_with_reliable_horizon":         len(horizon_elevations),
            "frames_horizon_excluded_instability":  len(eye_level_ys) - len(horizon_elevations) if horizon_valid else 0,
            "rolling_horizon_info": {
                "valid_sample_count":        rolling_horizon.get("valid_sample_count"),
                "total_sample_count":        rolling_horizon.get("total_sample_count"),
                "frames_with_horizon":       rolling_horizon.get("frames_with_horizon"),
                "frames_reliable":           rolling_horizon.get("frames_reliable"),
                "rolling_interval_s":        HORIZON_ROLLING_INTERVAL_S,
                "skip_s":                    HORIZON_SKIP_S,
                "instability_std_threshold": HORIZON_INSTABILITY_STD,
                "interp_max_gap_s":          HORIZON_INTERP_MAX_GAP_S,
                "method_note":               rolling_horizon.get("method_note"),
            },
            "eye_level_y_distribution":                    _distribution_summary(eye_level_ys),
            "eye_level_y_deviation_from_session_baseline": _distribution_summary(eye_level_deviations),
            "session_baseline_eye_y": round(baseline.baseline_eye_y, 3) if baseline.eye_y_valid and baseline.baseline_eye_y is not None else None,
            "pose_detection_rate": round(pose_rate, 3),
            "reliability": d2_reliability,
            "method_note": (
                "D2 positioning uses the rolling Hough horizon as the primary metric. "
                "The horizon Y is estimated from room geometry (Hough line detection on "
                "background regions) sampled every "
                f"{HORIZON_ROLLING_INTERVAL_S}s. elevation = horizon_y - eye_level_y. "
                f"at_patient_eye_level_rate = frames where |elevation| < {HORIZON_AT_LEVEL_THRESHOLD}. "
                f"above_patient_eye_level_rate = frames where elevation > {HORIZON_AT_LEVEL_THRESHOLD} "
                "(clinician above camera eye level — unfavourable). "
                "Frames where local horizon std exceeds "
                f"{HORIZON_INSTABILITY_STD} (patient head movement) are excluded "
                "and counted in frames_horizon_excluded_instability. "
                "reliability is based on the fraction of pose-detected frames with "
                "a reliable horizon estimate."
            ),
        }

        # ── D3: Posture ──────────────────────────────────────────────────────
        # Filter arm openness values above ARM_OPENNESS_MAX: these indicate a
        # wrist landmark has been detected far off-body (misdetection artefact)
        # and would inflate the mean and std. Excluded frames are counted
        # separately so the LLM scorer can see how many were dropped.
        arm_openness_raw = [
            f["positioning_and_posture"]["arm_openness"]
            for f in pose_frames
            if f.get("positioning_and_posture") and f["positioning_and_posture"].get("arm_openness") is not None
        ]
        arm_openness_vals    = [v for v in arm_openness_raw if v <= ARM_OPENNESS_MAX]
        arm_openness_outliers = len(arm_openness_raw) - len(arm_openness_vals)

        landmark_confs = [
            f["positioning_and_posture"]["landmark_confidence"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]
        arm_deviations = (
            [
                f["positioning_and_posture"]["baseline_deviation"]["arm_openness_deviation"]
                for f in pose_frames
                if f.get("positioning_and_posture")
                and f["positioning_and_posture"].get("baseline_deviation")
                and f["positioning_and_posture"]["baseline_deviation"].get("arm_openness_deviation") is not None
                and f["positioning_and_posture"].get("arm_openness") is not None
                and f["positioning_and_posture"]["arm_openness"] <= ARM_OPENNESS_MAX
            ]
            if baseline.valid and baseline.arm_valid else []
        )

        d3 = {
            "arm_openness_distribution":        _distribution_summary(arm_openness_vals),
            "baseline_arm_deviation":           _distribution_summary(arm_deviations),
            "landmark_confidence_distribution": _distribution_summary(landmark_confs),
            "arm_openness_outliers_excluded":   arm_openness_outliers,
            "pose_detection_rate":              round(pose_rate, 3),
            "reliability":                      _reliability_level(pose_rate),
            "method_note": (
                "Arm openness = wrist-to-wrist distance normalised by shoulder width. "
                "Values significantly below baseline suggest crossed arms. "
                f"Frames with arm_openness > {ARM_OPENNESS_MAX} are excluded from "
                "distributions as wrist landmark misdetection artefacts "
                "(see arm_openness_outliers_excluded). "
                "Camera roll from a seated patient is minimal; metric is reliable "
                "under normal conversational head movements."
            ),
        }

        # ── Item I: Professional behaviour ──────────────────────────────────
        d_i = {
            "gaze_on_target":            d1["gaze_on_target"],
            "positioning_at_eye_level":  d2["at_patient_eye_level_rate"],
            "positioning_above_eye_level": d2["above_patient_eye_level_rate"],
            "arm_openness_distribution": d3["arm_openness_distribution"],
            "overall_reliability":       _reliability_level(min(face_rate, pose_rate)),
            "method_note": (
                "Video-observable demeanour cues only. LUCAS Item I is primarily "
                "determined by verbal behaviour — video metrics are supporting "
                "evidence and should not be sole basis for scoring. "
                "positioning_at_eye_level / positioning_above_eye_level are from "
                "the rolling Hough horizon metric in D2."
            ),
        }

        return {
            "D1_eye_contact":                  d1,
            "D2_positioning":                  d2,
            "D3_posture":                      d3,
            "I_professional_behaviour_demeanour": d_i,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Build LLM-optimised output JSON
    # ═══════════════════════════════════════════════════════════════════════
    def _build_llm_output(self, nvb_metrics: Dict, baseline: PersonBaseline, metadata: Dict) -> Dict[str, Any]:
        output = {
            "analysis_metadata": {
                "video_fps":          metadata.get("video_fps"),
                "total_frames":       metadata.get("total_frames"),
                "frames_analyzed":    metadata.get("frames_analyzed"),
                "face_detection_rate": metadata.get("face_detection_rate"),
                "model":              metadata.get("model"),
                "workers":            metadata.get("workers"),
            },
            "person_baseline": baseline.to_dict(),
            "interpretation_guidance": {
                "camera_setup": (
                    "Egocentric (head-mounted) camera worn by the seated patient. "
                    "The camera optical axis approximates the patient's line of sight. "
                    "Camera pitch varies with patient head movement throughout the session."
                ),
                "note": (
                    "D1 uses iris landmark offsets (no baseline needed — self-referencing). "
                    "D2 uses rolling Hough horizon as the primary positioning metric. "
                    "D3 uses person-relative arm openness baseline. "
                    "Distribution summaries are provided instead of binary classifications."
                ),
                "reliability_scale": {
                    "high":     "Detection rate >= 75% — metric is trustworthy",
                    "moderate": "Detection rate 40-75% — interpret with caution",
                    "low":      "Detection rate < 40% — metric is unreliable",
                },
                "d1_gaze_interpretation": {
                    "on_target":    "iris_horizontal_offset ~0.5 AND iris_vertical_offset ~0.5 — clinician looking at patient",
                    "off_target":   "iris offset > 0.175 from centre — clinician looking away (notes, monitor, floor etc.)",
                    "missing_data": "face_detected=False frames are excluded from gaze metric entirely",
                },
                "d2_positioning_interpretation": {
                    "primary_method":             "rolling Hough horizon — uses room geometry to estimate camera eye level",
                    "elevation_positive":         "elevation > threshold → clinician above camera eye level → UNFAVOURABLE",
                    "elevation_near_zero":        "elevation within ±threshold → clinician at camera eye level → FAVOURABLE",
                    "at_patient_eye_level_rate":  f"proportion of reliable-horizon frames where |elevation| < {HORIZON_AT_LEVEL_THRESHOLD}",
                    "above_patient_eye_level_rate": f"proportion of reliable-horizon frames where elevation > {HORIZON_AT_LEVEL_THRESHOLD}",
                    "instability_exclusions":     "frames where patient head movement destabilises the horizon are excluded and counted separately",
                },
            },
        }
        output.update(nvb_metrics)
        return output

    # ═══════════════════════════════════════════════════════════════════════
    # Annotated video generation
    # ═══════════════════════════════════════════════════════════════════════
    def _generate_annotated_video(
        self, video_path, frame_data, cached_landmarks, output_path, cfg,
        rolling_horizon: Optional[Dict] = None,
    ):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video for annotation: {video_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*cfg.get("annotated_codec", "mp4v"))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            self.logger.error(f"Cannot open video writer: {output_path}")
            cap.release()
            return

        draw_landmarks  = cfg.get("annotated_draw_landmarks", True)
        feature_lookup  = {f["frame_idx"]: f for f in frame_data}
        horizon_lookup  = (rolling_horizon or {}).get("per_frame_lookup", {})
        frame_idx, last_features = 0, None

        self.logger.info(f"Rendering annotated video: {width}x{height} @ {fps} FPS ...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if draw_landmarks and frame_idx in cached_landmarks:
                self._draw_landmarks_cv2(frame, cached_landmarks[frame_idx], width, height)
            if frame_idx in feature_lookup:
                last_features = feature_lookup[frame_idx]
            if last_features:
                # Pass the per-frame horizon for this specific frame
                h_info        = horizon_lookup.get(frame_idx, {})
                frame_horizon = h_info.get("horizon_y") if h_info.get("horizon_reliable") else None
                self._draw_overlay(frame, last_features, width, height, horizon_y=frame_horizon)
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        self.logger.info(f"Annotated video: {frame_idx} frames → {output_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Landmark drawing helpers
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _draw_landmarks_on_frame(
        frame, landmarks, connections, connection_color, landmark_color,
        img_w, img_h, connection_thickness=2, landmark_radius=2, min_idx=0
    ):
        import cv2
        if landmarks is None or len(landmarks) == 0:
            return
        n = len(landmarks)
        for conn in connections:
            a, b = conn.start, conn.end
            if a < n and b < n:
                pt_a = (int(landmarks[a].x * img_w), int(landmarks[a].y * img_h))
                pt_b = (int(landmarks[b].x * img_w), int(landmarks[b].y * img_h))
                cv2.line(frame, pt_a, pt_b, connection_color, connection_thickness, cv2.LINE_AA)
        if landmark_color is not None:
            for idx in range(min_idx, n):
                pt = (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
                cv2.circle(frame, pt, landmark_radius, landmark_color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_landmarks_cv2(frame, cached, img_w, img_h):
        import cv2
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

        pose_lms = cached.get("pose_landmarks")
        if pose_lms and len(pose_lms) >= 33:
            VideoAnalysisStage._draw_landmarks_on_frame(
                frame, pose_lms,
                connections=PoseLandmarksConnections.POSE_LANDMARKS,
                connection_color=(200, 200, 200), landmark_color=(200, 200, 200),
                img_w=img_w, img_h=img_h, connection_thickness=1, landmark_radius=2, min_idx=11,
            )
        face_landmarks = cached.get("face_landmarks")
        if face_landmarks is not None and len(face_landmarks) >= 468:
            VideoAnalysisStage._draw_landmarks_on_frame(
                frame, face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                connection_color=(160, 160, 160), landmark_color=None,
                img_w=img_w, img_h=img_h, connection_thickness=1,
            )
        for hand_key in ("left_hand_landmarks", "right_hand_landmarks"):
            hand_lms = cached.get(hand_key)
            if hand_lms and len(hand_lms) >= 21:
                VideoAnalysisStage._draw_landmarks_on_frame(
                    frame, hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    connection_color=(200, 200, 200), landmark_color=(200, 200, 200),
                    img_w=img_w, img_h=img_h, connection_thickness=1, landmark_radius=2,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Overlay drawing
    # ═══════════════════════════════════════════════════════════════════════
    def _draw_overlay(self, frame, features, width, height, horizon_y: Optional[float] = None):
        """
        Left-panel overlay. horizon_y is the PER-FRAME interpolated value
        (None when the frame's local horizon is unreliable).
        """
        import cv2

        FONT      = cv2.FONT_HERSHEY_DUPLEX
        SCALE_HDR = 0.38
        SCALE_VAL = 0.44
        SCALE_LBL = 0.36
        THICK     = 1
        GAP       = 20
        GAP_S     = 14
        SEC       = 10
        DIM       = (110, 110, 110)
        BRIGHT    = (220, 220, 220)
        ALERT     = (160, 160, 160)
        PANEL_W   = 240

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (PANEL_W, height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # ── Per-frame horizon line (only when reliable) ──
        if horizon_y is not None:
            hy_px = int(horizon_y * height)
            dash_len, gap_len, x = 20, 10, PANEL_W
            while x < width:
                x_end = min(x + dash_len, width)
                cv2.line(frame, (x, hy_px), (x_end, hy_px), (160, 160, 160), 1, cv2.LINE_AA)
                x += dash_len + gap_len
            cv2.putText(frame, "cam level", (width - 110, hy_px - 4),
                        FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

        x, y = 12, 30

        def put(text, color=BRIGHT, scale=SCALE_VAL, gap=GAP):
            nonlocal y
            cv2.putText(frame, text, (x, y), FONT, scale, color, THICK, cv2.LINE_AA)
            y += gap

        def header(text):
            nonlocal y
            cv2.putText(frame, text, (x, y), FONT, SCALE_HDR, DIM, THICK, cv2.LINE_AA)
            y += GAP

        def spacer(n=SEC):
            nonlocal y
            y += n

        # D1
        header("D1  EYE CONTACT")
        ec = features.get("eye_contact")
        if ec:
            h_off  = ec.get("iris_horizontal_offset")
            v_off  = ec.get("iris_vertical_offset")
            on_tgt = ec.get("on_target", False)
            if h_off is not None:
                col   = BRIGHT if on_tgt else ALERT
                v_txt = f"  V={v_off:.2f}" if v_off is not None else ""
                put(f"H={h_off:.2f}{v_txt}  {'on' if on_tgt else 'off'}", color=col, gap=GAP_S)
            else:
                put("Iris not detected", color=DIM, gap=GAP_S)
        else:
            put("No face detected", color=DIM, gap=GAP_S)
        spacer()

        # D2
        header("D2  POSITIONING")
        pp = features.get("positioning_and_posture")
        if pp:
            eye_y = pp.get("eye_level_y")
            if horizon_y is not None and eye_y is not None:
                elev = horizon_y - eye_y
                if abs(elev) < HORIZON_AT_LEVEL_THRESHOLD:
                    pos_label, col = "at level", BRIGHT
                elif elev > HORIZON_AT_LEVEL_THRESHOLD:
                    pos_label, col = "above",    ALERT
                else:
                    pos_label, col = "below",    BRIGHT
                put(f"elev={elev:+.3f}  h={horizon_y:.3f}", color=DIM, gap=GAP_S)
                put(pos_label, color=col, gap=GAP_S)
            elif eye_y is not None:
                put(f"Eye Y {eye_y:.3f}  (no horizon)", color=DIM, gap=GAP_S)
            else:
                put("No eye landmarks", color=DIM, gap=GAP_S)
        else:
            put("No pose detected", color=DIM, gap=GAP_S)
        spacer()

        # D3
        header("D3  POSTURE")
        if pp:
            arm     = pp.get("arm_openness")
            arm_dev = (pp.get("baseline_deviation") or {}).get("arm_openness_deviation")
            if arm is not None:
                crossed = (arm_dev is not None and arm_dev < ARM_CROSSED_DEV) or arm < ARM_CROSSED_ABS
                col     = ALERT if crossed else BRIGHT
                dev_txt = f"  d={arm_dev:+.2f}" if arm_dev is not None else ""
                put(f"Arm open  {arm:.2f}{dev_txt}", color=col, gap=GAP_S)
        else:
            put("No pose detected", color=DIM, gap=GAP_S)
        spacer()

        # Item I
        header("Professional Behaviour")
        if ec and ec.get("on_target") is not None:
            put(f"Gaze  {'on target' if ec['on_target'] else 'off target'}", gap=GAP_S)
        if pp and pp.get("arm_openness") is not None:
            arm   = pp["arm_openness"]
            arm_d = (pp.get("baseline_deviation") or {}).get("arm_openness_deviation")
            crossed = (arm_d is not None and arm_d < ARM_CROSSED_DEV) or arm < ARM_CROSSED_ABS
            put(f"Arms  {'crossed' if crossed else 'open'}", gap=GAP_S)

        # Footer
        ts = features.get("timestamp_s", 0)
        cv2.putText(frame, f"{ts:.1f}s", (width - 70, height - 14), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        parts  = (["F"] if features.get("face_detected") else []) + \
                 (["P"] if features.get("pose_detected") else [])
        status = "+".join(parts) if parts else "none"
        cv2.putText(frame, f"Track: {status}", (x, height - 14), FONT, 0.40, DIM, 1, cv2.LINE_AA)