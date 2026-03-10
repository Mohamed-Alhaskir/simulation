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


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, NaN, and Infinity correctly."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.str_,)):
            return str(obj)
        return super().default(obj)

    def encode(self, obj):
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


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
# D1 — iris offset thresholds (normalised 0–1 within eye width/height)
# 0.5 = iris centered = looking straight at camera.
# Offsets beyond these thresholds indicate gaze directed away from camera.
IRIS_HORIZONTAL_THRESHOLD = 0.175   # |iris_offset_x - 0.5| > this → looking away horizontally
IRIS_VERTICAL_THRESHOLD   = 0.175   # |iris_offset_y - 0.5| > this → looking away vertically

SMILE_Z_THRESHOLD       = 1.0   # smile z-score above this → positive expression
PERIODICITY_THRESHOLD   = 0.3   # autocorrelation peak above this → repetitive
ARM_CROSSED_DEV         = -1.0  # arm openness deviation below → crossed (red)
ARM_CROSSED_DEV_WARN    = -0.5  # arm openness deviation below → warning
ARM_CROSSED_ABS         = 0.5   # absolute arm openness below → crossed

MIN_DOMAIN_SD = {
    "smile": 0.005,
}

# ---------------------------------------------------------------------------
# Horizon line estimation constants (D2 positioning)
# ---------------------------------------------------------------------------
# Hough line detection parameters for finding dominant horizontal lines
# in the room background (ceiling panels, door frames, wall edges).
HOUGH_RHO           = 1       # distance resolution (pixels)
HOUGH_THETA         = np.pi / 180  # angle resolution (1 degree)
HOUGH_THRESHOLD     = 80      # minimum votes to accept a line
HOUGH_MIN_LINE_LEN  = 80      # minimum line length (pixels)
HOUGH_MAX_LINE_GAP  = 20      # maximum gap to bridge within a line

# Only lines within this angular range from horizontal are considered
# (degrees from 0°). ±15° captures slightly tilted structural lines.
HORIZON_MAX_ANGLE_DEG = 15.0

# Number of frames to sample for session horizon estimation.
# Sampled evenly across first HORIZON_CALIBRATION_S seconds.
HORIZON_CALIBRATION_S      = 60    # seconds
HORIZON_SAMPLE_COUNT       = 20    # frames to sample
HORIZON_MIN_VALID_SAMPLES  = 5     # minimum needed for reliable estimate

# Fraction of frame width/height used as person-exclusion margin
# when masking the centre of the frame before Hough detection.
HORIZON_PERSON_MASK_W = 0.50   # exclude central 50% horizontally
HORIZON_PERSON_MASK_H = 0.60   # exclude central 60% vertically

# Fraction of frame height within which clinician is considered "at eye level"
# with the patient. Tighter = more precise but noisier.
# ±4% ≈ ±3cm height difference at typical clinical room scale.
# If estimated horizon_y is outside this range, treat as invalid.
# (Horizon should be somewhere in the frame, not at the very edge.)
HORIZON_VALID_RANGE = (0.05, 0.95)

# Fraction of frame height within which clinician is considered "at eye level"
# with the patient. ±4% ≈ ~3-4cm height difference at typical room scale.
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
            "5th": round(float(np.percentile(arr, 5)), 3),
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
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = len(values)
    return {k: round(v / total, 3) for k, v in counts.items()}


def _detect_periodicity(signal: List[float], fps: float) -> Dict[str, Any]:
    """
    Detect repetitive (periodic) patterns in a 1-D signal using
    autocorrelation. Only periods >= 0.3 seconds are considered.
    """
    if len(signal) < 20 or fps <= 0:
        return {
            "dominant_period_s": None,
            "periodicity_strength": 0.0,
            "is_repetitive": False,
        }

    arr = np.array(signal, dtype=float)
    smooth_window = max(3, int(fps * 0.2))
    if smooth_window < len(arr):
        kernel = np.ones(smooth_window) / smooth_window
        arr = np.convolve(arr, kernel, mode="valid")

    if len(arr) < 10:
        return {
            "dominant_period_s": None,
            "periodicity_strength": 0.0,
            "is_repetitive": False,
        }

    arr = arr - np.mean(arr)
    norm = np.dot(arr, arr)
    if norm < 1e-9:
        return {
            "dominant_period_s": None,
            "periodicity_strength": 0.0,
            "is_repetitive": False,
        }

    autocorr = np.correlate(arr, arr, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / norm

    MIN_PERIOD_S = 0.3
    min_lag = max(3, int(fps * MIN_PERIOD_S))

    peaks = []
    for i in range(min_lag, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append((i, autocorr[i]))

    if not peaks:
        return {
            "dominant_period_s": None,
            "periodicity_strength": 0.0,
            "is_repetitive": False,
        }

    best_lag, best_strength = max(peaks, key=lambda p: p[1])
    period_s = round(best_lag / fps, 2) if fps > 0 else None
    is_repetitive = bool(best_strength > PERIODICITY_THRESHOLD)

    return {
        "dominant_period_s": period_s,
        "periodicity_strength": round(float(best_strength), 3),
        "is_repetitive": is_repetitive,
    }


def _reliability_level(detection_rate: float) -> str:
    if detection_rate >= 0.75:
        return "high"
    elif detection_rate >= 0.40:
        return "moderate"
    else:
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
    Estimate the horizon Y position (normalised 0–1) from a single video
    frame using Hough probabilistic line detection on background regions.

    Scientific basis:
      In a fixed indoor room, structural lines (ceiling panels, door frames,
      wall edges, whiteboard edges) are approximately horizontal. These lines
      converge at the horizon — the Y level corresponding to the camera's
      optical axis. By finding the dominant cluster of near-horizontal lines
      in the background and taking their median Y intercept at frame centre,
      we obtain the camera's true eye-level in the scene.

      This is a standard vanishing-point technique used in autonomous
      driving (horizon detection for road geometry) and camera calibration.
      For an indoor clinical room with clear structural lines (as visible
      in the simulation room footage), it is reliable to within ~3–5% of
      frame height.

    Algorithm:
      1. Convert to grayscale and apply Canny edge detection
      2. Mask out the central person region to avoid body edges confounding
         the structural line detection
      3. Run probabilistic Hough transform to detect line segments
      4. Filter to near-horizontal lines (angle < HORIZON_MAX_ANGLE_DEG)
      5. For each line, compute its Y value at horizontal frame centre (x=0.5)
      6. Return the weighted median Y, weighted by line length

    Returns:
      Normalised Y of estimated horizon (0.0=top, 1.0=bottom), or None if
      insufficient lines found.
    """
    import cv2

    h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Canny edge detection — moderate thresholds to catch structural edges
    # without noise from textures
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Mask out the central person region
    mask = np.ones_like(edges)
    cx1 = int(w * (0.5 - person_mask_w / 2))
    cx2 = int(w * (0.5 + person_mask_w / 2))
    cy1 = int(h * (0.5 - person_mask_h / 2))
    cy2 = int(h * (0.5 + person_mask_h / 2))
    mask[cy1:cy2, cx1:cx2] = 0
    edges = edges * mask

    # Probabilistic Hough line detection
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

    # Filter to near-horizontal lines and compute their Y at frame centre
    horizon_ys = []
    weights    = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue  # vertical line — skip

        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        # Normalise to 0–90 range
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        if angle_deg > HORIZON_MAX_ANGLE_DEG:
            continue  # too steep — not a structural horizontal

        # Y value at horizontal frame centre (x = w/2)
        slope = dy / dx
        y_at_centre = y1 + slope * (w / 2 - x1)
        y_norm = y_at_centre / h

        if not (HORIZON_VALID_RANGE[0] <= y_norm <= HORIZON_VALID_RANGE[1]):
            continue

        line_length = math.sqrt(dx * dx + dy * dy)
        horizon_ys.append(y_norm)
        weights.append(line_length)

    if len(horizon_ys) < 3:
        return None

    # Weighted median — longer lines get more influence
    horizon_ys = np.array(horizon_ys)
    weights    = np.array(weights)

    # Sort by Y value
    sort_idx   = np.argsort(horizon_ys)
    sorted_ys  = horizon_ys[sort_idx]
    sorted_wts = weights[sort_idx]

    cumulative = np.cumsum(sorted_wts)
    median_idx = np.searchsorted(cumulative, cumulative[-1] / 2)
    horizon_y  = float(sorted_ys[min(median_idx, len(sorted_ys) - 1)])

    return round(horizon_y, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline computation
# ═══════════════════════════════════════════════════════════════════════════

class PersonBaseline:
    """
    Computes person-relative baselines from the first N seconds of video.
    Only sub-baselines with sufficient calibration frames are marked valid.
    Invalid baselines are not used — no artificial fallback SDs.

    NOTE: Gaze baseline (yaw/pitch) has been removed. D1 now uses iris
    landmark offsets which are self-referencing within the eye geometry —
    no external baseline is needed. The gaze baseline fields are retained
    as None so downstream code remains compatible.
    """

    MIN_CALIBRATION_FRAMES = 5

    def __init__(self, calibration_frames: List[Dict[str, Any]]):
        self.valid = len(calibration_frames) > 0

        # ── Gaze baseline — NOT USED for D1 (iris-based gaze is self-referencing) ──
        # Retained as invalid so any legacy code that checks baseline.gaze_valid
        # gets False and skips gracefully.
        self.gaze_valid = False
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_yaw_std = 1.0
        self.baseline_pitch_std = 1.0

        # ── Arm openness baseline ──
        arm_open = [
            f["positioning_and_posture"]["arm_openness"]
            for f in calibration_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("arm_openness") is not None
        ]
        self.arm_valid = len(arm_open) >= self.MIN_CALIBRATION_FRAMES
        if self.arm_valid:
            self.baseline_arm_openness = float(np.mean(arm_open))
        else:
            self.baseline_arm_openness = 0.8

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
                self.expression_valid = False
                self.baseline_smile_std = 1.0
            else:
                self.baseline_smile_std = smile_sd
        else:
            self.baseline_smile = 0.0
            self.baseline_smile_std = 1.0

        # ── Eye level baseline (D2) ──
        # Session-median eye_level_y from stable calibration frames.
        # Because the patient is seated on a chair and the camera is head-
        # mounted, camera pitch is approximately constant throughout the
        # session. The median over calibration frames gives a robust estimate
        # of the "neutral" (standing clinician) eye_level_y. Deviations below
        # this baseline indicate the clinician has crouched to patient level.
        eye_ys = [
            f["positioning_and_posture"]["eye_level_y"]
            for f in calibration_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_y") is not None
        ]
        self.baseline_eye_y = float(np.median(eye_ys)) if eye_ys else None
        self.eye_y_valid = len(eye_ys) >= self.MIN_CALIBRATION_FRAMES

        # ── Calibration quality report ──
        self.calibration_quality = {
            "total_calibration_frames": len(calibration_frames),
            "posture_arm_frames": len(arm_open),
            "expression_frames": len(smiles),
            "eye_level_y_frames": len(eye_ys),
            "arm_baseline_valid": self.arm_valid,
            "expression_baseline_valid": self.expression_valid,
            "eye_y_baseline_valid": self.eye_y_valid,
            "minimum_required": self.MIN_CALIBRATION_FRAMES,
        }

    def posture_deviation(self, arm_openness: float) -> Optional[Dict[str, Any]]:
        if self.arm_valid and arm_openness is not None:
            arm_dev = arm_openness - self.baseline_arm_openness
        else:
            arm_dev = None
        return {
            "arm_openness_deviation": round(arm_dev, 3) if arm_dev is not None else None,
        }

    def expression_deviation(self, smile: float) -> Optional[Dict[str, float]]:
        if not self.expression_valid:
            return None
        smile_dev = smile - self.baseline_smile
        smile_z = smile_dev / self.baseline_smile_std if self.baseline_smile_std > 0 else 0
        return {
            "smile_deviation": round(smile_dev, 3),
            "smile_z_score": round(smile_z, 2),
        }

    def eye_level_deviation(self, eye_level_y: float) -> Optional[Dict[str, Any]]:
        """
        Compute deviation of clinician eye Y from session baseline.
        Positive deviation (eye_level_y > baseline) means clinician's eyes
        are lower in frame than baseline → clinician has crouched → GOOD.
        Negative deviation means clinician is higher than baseline → standing
        over patient → less favourable for paediatric consultation.
        """
        if not self.eye_y_valid or self.baseline_eye_y is None:
            return None
        deviation = eye_level_y - self.baseline_eye_y
        return {
            "eye_level_y_deviation": round(deviation, 3),
            "below_baseline": deviation > 0,  # True = crouched relative to baseline
        }

    def to_dict(self) -> Dict[str, Any]:
        if not self.valid:
            return {"valid": False, "calibration_quality": self.calibration_quality if hasattr(self, 'calibration_quality') else {}}
        return {
            "valid": True,
            "calibration_quality": self.calibration_quality,
            "posture": {
                "arm_valid": self.arm_valid,
                "resting_arm_openness": round(self.baseline_arm_openness, 3) if self.arm_valid else None,
            },
            "expression": {
                "valid": self.expression_valid,
                "resting_smile": round(self.baseline_smile, 4),
            },
            "positioning": {
                "eye_y_valid": self.eye_y_valid,
                "session_median_eye_level_y": round(self.baseline_eye_y, 3) if self.baseline_eye_y is not None else None,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main stage class
# ═══════════════════════════════════════════════════════════════════════════

class VideoAnalysisStage(BaseStage):
    """LUCAS-aligned NVB video analysis stage."""

    # ── Face landmark indices ──
    NOSE_TIP = 1
    LEFT_EYE_INNER = 133; RIGHT_EYE_INNER = 362
    LEFT_EYE_OUTER = 33;  RIGHT_EYE_OUTER = 263
    CHIN = 152; FOREHEAD = 10
    LEFT_MOUTH = 61; RIGHT_MOUTH = 291

    # ── Iris landmark indices (MediaPipe 478-point model) ──
    # Each iris: centre + 4 cardinal edge points
    LEFT_IRIS_CENTER  = 468
    LEFT_IRIS_RIGHT   = 469  # rightmost point of left iris
    LEFT_IRIS_TOP     = 470
    LEFT_IRIS_LEFT    = 471  # leftmost point of left iris
    LEFT_IRIS_BOTTOM  = 472
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_RIGHT  = 474
    RIGHT_IRIS_TOP    = 475
    RIGHT_IRIS_LEFT   = 476
    RIGHT_IRIS_BOTTOM = 477

    # ── Eye corner indices used for iris offset normalisation ──
    # LEFT eye: inner corner = 133, outer corner = 33, top = 159, bottom = 145
    # RIGHT eye: inner corner = 362, outer corner = 263, top = 386, bottom = 374
    LEFT_EYE_TOP    = 159; LEFT_EYE_BOTTOM  = 145
    RIGHT_EYE_TOP   = 386; RIGHT_EYE_BOTTOM = 374

    # ── Pose landmark indices ──
    POSE_NOSE = 0
    POSE_LEFT_SHOULDER = 11; POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_ELBOW = 13;    POSE_RIGHT_ELBOW = 14
    POSE_LEFT_WRIST = 15;    POSE_RIGHT_WRIST = 16
    POSE_LEFT_HIP = 23;      POSE_RIGHT_HIP = 24

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

        sample_fps = cfg.get("sample_fps", 2)

        results = self._analyze_video(video_path, sample_fps, cfg)
        self.logger.info(
            f"Analyzed {results['metadata']['frames_analyzed']} frames "
            f"({results['metadata']['faces_detected']} with face detected)"
        )

        frame_data = results["frame_data"]
        video_fps = results["metadata"]["video_fps"]

        calibration_s = cfg.get("calibration_seconds", self.DEFAULT_CALIBRATION_SECONDS)
        calibration_frames = [f for f in frame_data if f["timestamp_s"] <= calibration_s]
        baseline = PersonBaseline(calibration_frames)
        self.logger.info(
            f"Baseline computed from {len(calibration_frames)} calibration "
            f"frames ({calibration_s}s window). Valid: {baseline.valid}"
        )

        if baseline.valid:
            for f in frame_data:
                self._enrich_frame_with_baseline(f, baseline)

        # ── D2: Estimate session horizon from room geometry ──
        # This gives the true camera eye-level Y in normalised frame coords,
        # derived from structural horizontal lines (ceiling, door frames, walls).
        # Required because camera pitch is fixed but unknown — a clinician whose
        # face appears "centred" in frame may still be above patient eye level
        # if the camera is tilted upward.
        horizon_info = self._estimate_session_horizon(
            video_path=video_path,
            video_fps=video_fps,
            calibration_s=cfg.get("calibration_seconds", self.DEFAULT_CALIBRATION_SECONDS),
        )
        self.logger.info(
            f"[D2 Horizon] valid={horizon_info['horizon_valid']} "
            f"y={horizon_info.get('horizon_y')} "
            f"reliability={horizon_info.get('reliability')}"
        )

        nvb_metrics = self._compute_lucas_nvb_metrics(
            frame_data, baseline, video_fps, horizon_info
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
                horizon_y=horizon_info.get("horizon_y") if horizon_info.get("horizon_valid") else None,
            )

        lucas_nvb_output = self._build_llm_output(
            nvb_metrics=nvb_metrics,
            baseline=baseline,
            metadata=results["metadata"],
        )

        features_path = output_dir / "video_features.json"
        with open(features_path, "w") as f:
            json.dump(lucas_nvb_output, f, indent=2, ensure_ascii=False, cls=_NumpySafeEncoder)

        ctx["artifacts"]["video_features"] = lucas_nvb_output
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

        inventory_quadrant = inventory.get("video_analysis_quadrant")
        if inventory_quadrant:
            if inventory_quadrant in quadrants:
                self.logger.info(
                    f"Video analysis quadrant from inventory: '{inventory_quadrant}'"
                )
                return quadrants[inventory_quadrant]
            self.logger.warning(
                f"inventory.video_analysis_quadrant='{inventory_quadrant}' not found "
                f"in quadrants {list(quadrants.keys())}; falling back."
            )

        preferred_quadrant = cfg.get("preferred_quadrant")
        if preferred_quadrant and preferred_quadrant in quadrants:
            self.logger.info(
                f"Video analysis quadrant from config: '{preferred_quadrant}'"
            )
            return quadrants[preferred_quadrant]

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
    # D2 — Session horizon estimation from room geometry
    # ═══════════════════════════════════════════════════════════════════════
    def _estimate_session_horizon(
        self,
        video_path: str,
        video_fps: float,
        calibration_s: float = HORIZON_CALIBRATION_S,
        sample_count: int = HORIZON_SAMPLE_COUNT,
    ) -> Dict[str, Any]:
        """
        Estimate the session-level horizon Y by sampling frames from the
        calibration window and running Hough-based horizon detection on each.

        The session horizon is the robust median across valid sample frames.
        It represents the true camera eye-level in the scene — the Y level
        in normalised frame coordinates that corresponds to "same height as
        the camera" (= same height as the seated patient).

        A clinician whose eye_level_y is ABOVE the horizon (lower Y value)
        is standing taller than the patient.
        A clinician whose eye_level_y is AT or BELOW the horizon (higher Y)
        is at or below patient eye level — the desired positioning for
        paediatric consultations.

        Returns a dict with:
          horizon_y         — session median horizon (normalised, 0=top 1=bottom)
          horizon_valid     — True if enough samples were found
          sample_count      — number of frames sampled
          valid_count       — number of frames with successful detection
          horizon_std       — SD across samples (stability indicator)
          reliability       — "high" / "moderate" / "low"
          method_note       — human-readable explanation
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"horizon_valid": False, "horizon_y": None,
                    "method_note": "Could not open video for horizon estimation."}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        calib_frames = min(total_frames, int(calibration_s * video_fps))

        # Sample evenly across calibration window
        sample_indices = np.linspace(0, calib_frames - 1, sample_count, dtype=int)

        horizon_estimates = []
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            h_y = _estimate_horizon_y_from_frame(frame)
            if h_y is not None:
                horizon_estimates.append(h_y)

        cap.release()

        valid_count = len(horizon_estimates)

        if valid_count < HORIZON_MIN_VALID_SAMPLES:
            self.logger.warning(
                f"[D2 Horizon] Only {valid_count}/{sample_count} valid samples — "
                f"horizon estimate unreliable. Room may lack clear horizontal lines."
            )
            return {
                "horizon_valid": False,
                "horizon_y": None,
                "sample_count": sample_count,
                "valid_count": valid_count,
                "reliability": "low",
                "method_note": (
                    "Horizon estimation failed: insufficient horizontal structural "
                    "lines detected in background. D2 positioning falls back to "
                    "session-median eye_level_y baseline."
                ),
            }

        arr = np.array(horizon_estimates)
        session_horizon_y = float(np.median(arr))
        horizon_std       = float(np.std(arr))

        # Reliability based on consistency (low std = stable detection)
        if horizon_std < 0.04 and valid_count >= HORIZON_MIN_VALID_SAMPLES * 2:
            reliability = "high"
        elif horizon_std < 0.08:
            reliability = "moderate"
        else:
            reliability = "low"

        self.logger.info(
            f"[D2 Horizon] Session horizon Y = {session_horizon_y:.3f} "
            f"(std={horizon_std:.3f}, {valid_count}/{sample_count} valid, "
            f"reliability={reliability})"
        )

        return {
            "horizon_valid": True,
            "horizon_y": round(session_horizon_y, 4),
            "horizon_std": round(horizon_std, 4),
            "sample_count": sample_count,
            "valid_count": valid_count,
            "reliability": reliability,
            "method_note": (
                "Horizon estimated via Hough probabilistic line detection on "
                "background regions (person masked out). Near-horizontal lines "
                "(< 15 deg) from structural room features (ceiling panels, door "
                "frames, wall edges) are detected; their weighted-median Y at "
                "frame centre gives the camera eye-level in the scene. "
                f"Session median = {session_horizon_y:.3f} "
                f"(std={horizon_std:.3f} across {valid_count} samples)."
            ),
        }

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
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_interval = 1

        available_cores = cpu_count() or 4
        num_workers = cfg.get("num_workers", min(available_cores, 40))
        segment_size = max(1, total_frames // num_workers)

        self.logger.info(
            f"[Parallel] {video_fps:.1f} FPS | {total_frames} frames | "
            f"interval={frame_interval} | workers={num_workers}"
        )

        proto, model = _ensure_dnn_model()
        model_paths = {
            "face": _ensure_model("face"),
            "pose": _ensure_model("pose"),
            "hand": _ensure_model("hand"),
            "dnn_proto": proto,
            "dnn_model": model,
        }
        segments = []
        for i in range(num_workers):
            start = i * segment_size
            end = (i + 1) * segment_size if i < num_workers - 1 else total_frames
            segments.append((video_path, start, end, frame_interval, cfg, model_paths))

        ctx_mp = get_context("spawn")
        with ctx_mp.Pool(processes=num_workers) as pool:
            results = pool.map(partial(self._process_video_segment), segments)

        frame_data, cached_landmarks = [], {}
        faces_detected, frames_analyzed = 0, 0
        for r in sorted(results, key=lambda x: x["segment_start"]):
            frame_data.extend(r["frame_data"])
            faces_detected += r["faces_detected"]
            frames_analyzed += r["frames_analyzed"]
            cached_landmarks.update(r.get("cached_landmarks", {}))

        frame_data.sort(key=lambda x: x["frame_idx"])

        return {
            "metadata": {
                "video_path": video_path,
                "video_fps": video_fps,
                "total_frames": total_frames,
                "sample_fps": sample_fps,
                "frames_analyzed": frames_analyzed,
                "faces_detected": faces_detected,
                "face_detection_rate": round(faces_detected / frames_analyzed, 3)
                if frames_analyzed > 0 else 0,
                "model": "mediapipe_tasks_face+pose+hand_parallel",
                "workers": num_workers,
            },
            "frame_data": frame_data,
            "cached_landmarks": cached_landmarks,
        }

    def _process_video_segment(self, args):
        """
        Process a segment of video frames in a worker process.

        Two-stage face detection:
          Stage 1 — OpenCV DNN res10 SSD on full frame → bounding box
          Stage 2 — MediaPipe FaceLandmarker on padded, upscaled face crop
          Fallback — MediaPipe on full frame if Stage 1 finds nothing

        Landmark coordinates are remapped from crop-normalised space back
        to full-frame normalised space before storage.
        """
        import os, cv2, mediapipe as mp

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

        video_path, start_frame, end_frame, frame_interval, cfg, model_paths = args

        face_model_path = model_paths["face"]
        pose_model_path = model_paths["pose"]
        hand_model_path = model_paths["hand"]
        dnn_proto = model_paths["dnn_proto"]
        dnn_model = model_paths["dnn_model"]

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        det_conf = cfg.get("detection_confidence", 0.4)
        trk_conf = cfg.get("tracking_confidence", 0.4)
        dnn_conf = cfg.get("dnn_confidence", 0.3)
        face_crop_size = cfg.get("face_crop_size", 320)
        face_pad = cfg.get("face_pad", 0.4)
        upscale_factor = cfg.get("upscale_factor", 1.0)
        do_upscale = upscale_factor > 1.0
        dnn_interval = cfg.get("dnn_interval", 1)

        dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)

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

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_idx = start_frame
        frames_analyzed, faces_detected = 0, 0
        frame_data, cached_landmarks = [], {}

        _last_dnn_bbox = None
        _last_dnn_conf = 0.0

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
                    timestamp_s = frame_idx / video_fps

                    orig_h, orig_w = frame.shape[:2]
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    _run_dnn = (frame_idx % dnn_interval == 0)

                    if _run_dnn:
                        blob = cv2.dnn.blobFromImage(
                            frame, scalefactor=1.0, size=(300, 300),
                            mean=(104.0, 177.0, 123.0),
                            swapRB=False, crop=False,
                        )
                        dnn_net.setInput(blob)
                        detections = dnn_net.forward()

                        best_bbox_norm = None
                        best_dnn_conf = 0.0
                        for d in range(detections.shape[2]):
                            confidence = float(detections[0, 0, d, 2])
                            if confidence > dnn_conf and confidence > best_dnn_conf:
                                best_dnn_conf = confidence
                                best_bbox_norm = detections[0, 0, d, 3:7]

                        if best_bbox_norm is not None:
                            _last_dnn_bbox = best_bbox_norm
                            _last_dnn_conf = best_dnn_conf
                    else:
                        best_bbox_norm = _last_dnn_bbox
                        best_dnn_conf = _last_dnn_conf

                    # Stage 2a: MediaPipe on face crop
                    face_result = None
                    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, orig_w, orig_h
                    used_crop = False

                    if best_bbox_norm is not None:
                        bx1 = int(best_bbox_norm[0] * orig_w)
                        by1 = int(best_bbox_norm[1] * orig_h)
                        bx2 = int(best_bbox_norm[2] * orig_w)
                        by2 = int(best_bbox_norm[3] * orig_h)
                        bw = max(bx2 - bx1, 1)
                        bh = max(by2 - by1, 1)

                        pad_x = int(bw * face_pad)
                        pad_y = int(bh * face_pad)
                        cx1 = max(0, bx1 - pad_x)
                        cy1 = max(0, by1 - pad_y)
                        cx2 = min(orig_w, bx2 + pad_x)
                        cy2 = min(orig_h, by2 + pad_y)

                        crop_x1, crop_y1 = cx1, cy1
                        crop_x2, crop_y2 = cx2, cy2
                        crop_w = cx2 - cx1
                        crop_h = cy2 - cy1

                        if crop_w > 10 and crop_h > 10:
                            face_crop = rgb_full[cy1:cy2, cx1:cx2]
                            face_crop_resized = cv2.resize(
                                face_crop,
                                (face_crop_size, face_crop_size),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            mp_face_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=face_crop_resized,
                            )
                            face_result = face_lm.detect(mp_face_image)
                            used_crop = True

                    # Stage 2b: Fallback to full frame
                    if face_result is None or not face_result.face_landmarks:
                        full_for_face = rgb_full
                        if do_upscale:
                            full_for_face = cv2.resize(
                                rgb_full,
                                (int(orig_w * upscale_factor), int(orig_h * upscale_factor)),
                                interpolation=cv2.INTER_CUBIC,
                            )
                        mp_full_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=full_for_face,
                        )
                        face_result = face_lm.detect(mp_full_image)
                        crop_x1, crop_y1 = 0, 0
                        crop_x2, crop_y2 = orig_w, orig_h
                        used_crop = False

                    face_confirmed = bool(
                        face_result and face_result.face_landmarks
                    )

                    if not face_confirmed:
                        _last_dnn_bbox = None
                        _last_dnn_conf = 0.0

                        cached_landmarks[frame_idx] = {
                            "face_landmarks": None,
                            "pose_landmarks": None,
                            "left_hand_landmarks": None,
                            "right_hand_landmarks": None,
                            "face_crop": None,
                        }
                        frame_data.append({
                            "timestamp_s": round(timestamp_s, 2),
                            "frame_idx": frame_idx,
                            "dnn_face_confidence": round(best_dnn_conf, 3),
                            "used_face_crop": used_crop,
                            "face_detected": False,
                            "pose_detected": False,
                            "eye_contact": None,
                            "facial_expression": None,
                            "positioning_and_posture": None,
                            "gestures": None,
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
                        pose_img_w = orig_w
                        pose_img_h = orig_h

                    mp_pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=full_rgb_pose)
                    pose_result = pose_lm.detect_for_video(mp_pose_image, timestamp_ms)
                    hand_result = hand_lm.detect_for_video(mp_pose_image, timestamp_ms)

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
                        "face_landmarks": face_landmarks,
                        "pose_landmarks": pose_landmarks,
                        "left_hand_landmarks": left_hand_lms,
                        "right_hand_landmarks": right_hand_lms,
                        "face_crop": (crop_x1, crop_y1, crop_x2, crop_y2) if used_crop else None,
                    }

                    frame_features = {
                        "timestamp_s": round(timestamp_s, 2),
                        "frame_idx": frame_idx,
                        "dnn_face_confidence": round(best_dnn_conf, 3),
                        "used_face_crop": used_crop,
                    }

                    faces_detected += 1
                    frame_features["face_detected"] = True
                    frame_features["eye_contact"] = self._extract_iris_gaze(
                        face_landmarks, orig_w, orig_h
                    )
                    frame_features["facial_expression"] = self._extract_facial_expression(
                        blendshapes, face_landmarks, orig_w, orig_h
                    )

                    if pose_landmarks and len(pose_landmarks) > 0:
                        frame_features["pose_detected"] = True
                        frame_features["positioning_and_posture"] = self._extract_positioning_posture(
                            pose_landmarks,
                            face_landmarks=face_landmarks,
                            img_w=pose_img_w, img_h=pose_img_h,
                        )
                    else:
                        frame_features["pose_detected"] = False
                        frame_features["positioning_and_posture"] = None

                    has_left = left_hand_lms and len(left_hand_lms) > 0
                    has_right = right_hand_lms and len(right_hand_lms) > 0
                    if has_left or has_right:
                        frame_features["gestures"] = self._extract_gestures(
                            left_hand_lms if has_left else None,
                            right_hand_lms if has_right else None,
                            pose_landmarks, pose_img_w, pose_img_h,
                        )
                    else:
                        frame_features["gestures"] = None

                    frame_data.append(frame_features)
                    frames_analyzed += 1
                  except Exception as _frame_err:
                    frame_data.append({
                        "timestamp_s": round(frame_idx / video_fps, 2),
                        "frame_idx": frame_idx,
                        "face_detected": False,
                        "pose_detected": False,
                        "eye_contact": None,
                        "facial_expression": None,
                        "positioning_and_posture": None,
                        "gestures": None,
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
            "segment_start": start_frame,
            "frames_analyzed": frames_analyzed,
            "faces_detected": faces_detected,
            "frame_data": frame_data,
            "cached_landmarks": cached_landmarks,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Face landmark coordinate remapping (crop → full frame)
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _remap_face_landmarks(
        landmarks,
        crop_x1: int, crop_y1: int,
        crop_x2: int, crop_y2: int,
        orig_w: int, orig_h: int,
    ):
        from types import SimpleNamespace

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w <= 0 or crop_h <= 0:
            return landmarks

        remapped = []
        for lm in landmarks:
            px_full_x = (lm.x * crop_w) + crop_x1
            px_full_y = (lm.y * crop_h) + crop_y1
            remapped.append(SimpleNamespace(
                x=px_full_x / orig_w,
                y=px_full_y / orig_h,
                z=getattr(lm, "z", 0.0),
                visibility=getattr(lm, "visibility", 1.0),
            ))
        return remapped

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline enrichment
    # ═══════════════════════════════════════════════════════════════════════
    def _enrich_frame_with_baseline(self, frame: Dict, baseline: PersonBaseline):
        # D1 — iris gaze needs no baseline enrichment (self-referencing metric)

        pp = frame.get("positioning_and_posture")
        if pp:
            arm = pp.get("arm_openness")
            pp["baseline_deviation"] = baseline.posture_deviation(arm)
            # D2 — enrich eye_level_y with session baseline deviation
            eye_y = pp.get("eye_level_y")
            if eye_y is not None:
                pp["eye_level_baseline_deviation"] = baseline.eye_level_deviation(eye_y)

        fe = frame.get("facial_expression")
        if fe and fe.get("blendshape_smile") is not None:
            fe["baseline_deviation"] = baseline.expression_deviation(fe["blendshape_smile"])

    # ═══════════════════════════════════════════════════════════════════════
    # D1 — Eye-contact via iris landmark gaze direction
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_iris_gaze(self, landmarks, img_w: int, img_h: int) -> Optional[Dict[str, Any]]:
        """
        Estimate gaze direction from iris landmark position relative to eye
        corners. This is the correct metric for a head-mounted egocentric
        camera because it measures where the eyes are actually pointing
        within the face, independent of head orientation.

        Scientific basis:
          The iris center position relative to the eye width gives a
          normalised horizontal offset:
            0.0 = iris at the inner corner (looking inward / toward nose)
            0.5 = iris centered = looking straight at camera = AT the patient
            1.0 = iris at the outer corner (looking outward / away)
          Same logic applies vertically.

        Both eyes are averaged. A frame is "on target" when both horizontal
        and vertical offsets are within threshold of 0.5.

        Requires MediaPipe 478-landmark model (iris landmarks 468-477).
        Falls back to None if iris landmarks are absent (older 468-pt model).

        NOTE: Face non-detection (eye_contact = None in frame_data) means
        the face was not visible — this is treated as MISSING DATA, not as
        "not looking". The patient may have turned their head. Only frames
        with a detected face yield a valid gaze measurement.
        """
        if landmarks is None:
            return None

        # Need 478 landmarks for iris support
        if len(landmarks) < 478:
            return None

        def lm(idx):
            l = landmarks[idx]
            return np.array([l.x, l.y])

        # ── Left eye iris offset ──
        l_iris   = lm(self.LEFT_IRIS_CENTER)
        l_inner  = lm(self.LEFT_EYE_INNER)   # 133
        l_outer  = lm(self.LEFT_EYE_OUTER)   # 33
        l_top    = lm(self.LEFT_EYE_TOP)     # 159
        l_bottom = lm(self.LEFT_EYE_BOTTOM)  # 145

        l_eye_width  = np.linalg.norm(l_outer - l_inner)
        l_eye_height = np.linalg.norm(l_bottom - l_top)

        if l_eye_width < 1e-4 or l_eye_height < 1e-4:
            left_h_offset, left_v_offset = None, None
        else:
            # Project iris onto the eye axis vectors
            eye_h_axis = (l_outer - l_inner) / l_eye_width
            eye_v_axis = (l_bottom - l_top) / l_eye_height
            iris_rel = l_iris - l_inner
            left_h_offset = float(np.dot(iris_rel, eye_h_axis) / l_eye_width)
            iris_rel_v = l_iris - l_top
            left_v_offset = float(np.dot(iris_rel_v, eye_v_axis) / l_eye_height)

        # ── Right eye iris offset ──
        r_iris   = lm(self.RIGHT_IRIS_CENTER)
        r_inner  = lm(self.RIGHT_EYE_INNER)   # 362
        r_outer  = lm(self.RIGHT_EYE_OUTER)   # 263
        r_top    = lm(self.RIGHT_EYE_TOP)     # 386
        r_bottom = lm(self.RIGHT_EYE_BOTTOM)  # 374

        r_eye_width  = np.linalg.norm(r_outer - r_inner)
        r_eye_height = np.linalg.norm(r_bottom - r_top)

        if r_eye_width < 1e-4 or r_eye_height < 1e-4:
            right_h_offset, right_v_offset = None, None
        else:
            eye_h_axis = (r_outer - r_inner) / r_eye_width
            eye_v_axis = (r_bottom - r_top) / r_eye_height
            iris_rel = r_iris - r_inner
            right_h_offset = float(np.dot(iris_rel, eye_h_axis) / r_eye_width)
            iris_rel_v = r_iris - r_top
            right_v_offset = float(np.dot(iris_rel_v, eye_v_axis) / r_eye_height)

        # ── Average both eyes ──
        h_offsets = [o for o in [left_h_offset, right_h_offset] if o is not None]
        v_offsets = [o for o in [left_v_offset, right_v_offset] if o is not None]

        if not h_offsets:
            return None

        avg_h = float(np.mean(h_offsets))
        avg_v = float(np.mean(v_offsets)) if v_offsets else None

        # on_target: iris centered horizontally AND vertically
        h_on_target = abs(avg_h - 0.5) < IRIS_HORIZONTAL_THRESHOLD
        v_on_target = (abs(avg_v - 0.5) < IRIS_VERTICAL_THRESHOLD) if avg_v is not None else True
        on_target = h_on_target and v_on_target

        return {
            "iris_horizontal_offset": round(avg_h, 3),
            "iris_vertical_offset": round(avg_v, 3) if avg_v is not None else None,
            "on_target": on_target,
            "left_iris_h_offset": round(left_h_offset, 3) if left_h_offset is not None else None,
            "right_iris_h_offset": round(right_h_offset, 3) if right_h_offset is not None else None,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # D4 — Facial expressions (blendshape-based, smile only)
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_facial_expression(
        self,
        blendshapes: Optional[Dict[str, float]],
        face_landmarks,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        if blendshapes:
            smile_l = blendshapes.get("mouthSmileLeft", 0.0)
            smile_r = blendshapes.get("mouthSmileRight", 0.0)
            smile = (smile_l + smile_r) / 2
            return {
                "source": "blendshapes",
                "blendshape_smile": round(smile, 4),
            }
        else:
            # Fallback: landmark geometry
            def lm_2d(idx):
                lm = face_landmarks[idx]
                return np.array([lm.x * img_w, lm.y * img_h])

            left_mouth  = lm_2d(self.LEFT_MOUTH)
            right_mouth = lm_2d(self.RIGHT_MOUTH)
            nose_tip    = lm_2d(self.NOSE_TIP)
            chin        = lm_2d(self.CHIN)
            face_height = np.linalg.norm(chin - nose_tip)

            mouth_center = (left_mouth + right_mouth) / 2
            smile_score = (
                (mouth_center[1] - left_mouth[1])
                + (mouth_center[1] - right_mouth[1])
            ) / 2
            normalized_smile = smile_score / face_height if face_height > 0 else 0

            return {
                "source": "landmark_geometry_fallback",
                "blendshape_smile": round(float(normalized_smile), 4),
                "reliability_note": (
                    "Blendshapes unavailable; using landmark geometry fallback "
                    "which is less reliable."
                ),
            }

    # ═══════════════════════════════════════════════════════════════════════
    # D2 — Positioning  &  D3 — Posture
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_positioning_posture(
        self, pose_landmarks, face_landmarks=None, img_w=None, img_h=None
    ) -> Dict[str, Any]:
        def plm(idx):
            lm = pose_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z]), getattr(lm, "visibility", 1.0)

        left_shoulder,  ls_vis = plm(self.POSE_LEFT_SHOULDER)
        right_shoulder, rs_vis = plm(self.POSE_RIGHT_SHOULDER)
        left_hip,       lh_vis = plm(self.POSE_LEFT_HIP)
        right_hip,      rh_vis = plm(self.POSE_RIGHT_HIP)
        left_wrist,     lw_vis = plm(self.POSE_LEFT_WRIST)
        right_wrist,    rw_vis = plm(self.POSE_RIGHT_WRIST)

        MIN_SHOULDER_WIDTH = 0.05
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_width_valid = shoulder_width >= MIN_SHOULDER_WIDTH

        MAX_ARM_OPENNESS = 5.0
        wrist_distance = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
        if shoulder_width_valid:
            arm_openness = min(wrist_distance / shoulder_width, MAX_ARM_OPENNESS)
        else:
            arm_openness = None

        avg_vis = np.mean([ls_vis, rs_vis, lh_vis, rh_vis])

        # Eye level Y for D2 positioning
        eye_level_y = None
        if face_landmarks and len(face_landmarks) > max(self.LEFT_EYE_INNER, self.RIGHT_EYE_INNER):
            left_eye_y  = face_landmarks[self.LEFT_EYE_INNER].y
            right_eye_y = face_landmarks[self.RIGHT_EYE_INNER].y
            eye_level_y = round(float((left_eye_y + right_eye_y) / 2), 3)

        return {
            "arm_openness": round(float(arm_openness), 3) if arm_openness is not None else None,
            "shoulder_width_valid": shoulder_width_valid,
            "eye_level_y": eye_level_y,
            "landmark_confidence": round(float(avg_vis), 3),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # D5 — Gestures & mannerisms
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_gestures(
        self, left_hand_lms, right_hand_lms, pose_landmarks, img_w, img_h
    ) -> Dict[str, Any]:
        hands_info = []
        for label, hand_lms in [("left", left_hand_lms), ("right", right_hand_lms)]:
            if hand_lms is None or len(hand_lms) == 0:
                continue
            wrist      = hand_lms[0]
            middle_tip = hand_lms[12]
            cy = np.mean([lm.y for lm in hand_lms])
            spread = np.linalg.norm(
                np.array([middle_tip.x - wrist.x, middle_tip.y - wrist.y])
            )

            position = "unknown"
            if pose_landmarks and len(pose_landmarks) > self.POSE_RIGHT_HIP:
                shoulder_y = (
                    pose_landmarks[self.POSE_LEFT_SHOULDER].y
                    + pose_landmarks[self.POSE_RIGHT_SHOULDER].y
                ) / 2
                hip_y = (
                    pose_landmarks[self.POSE_LEFT_HIP].y
                    + pose_landmarks[self.POSE_RIGHT_HIP].y
                ) / 2
                if cy < shoulder_y:
                    position = "above_shoulders"
                elif cy < hip_y:
                    position = "torso_level"
                else:
                    position = "below_hips"

            hands_info.append({
                "hand": label,
                "spread": round(float(spread), 3),
                "center_y": round(float(cy), 3),
                "position": position,
            })

        return {
            "num_hands_visible": len(hands_info),
            "hands": hands_info,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate LUCAS NVB metrics
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_lucas_nvb_metrics(
        self,
        frame_data: List[Dict],
        baseline: PersonBaseline,
        video_fps: float,
        horizon_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if not frame_data:
            return {}

        total = len(frame_data)
        face_frames = [f for f in frame_data if f.get("face_detected")]
        pose_frames = [f for f in frame_data if f.get("pose_detected")]
        face_rate = len(face_frames) / total if total > 0 else 0
        pose_rate = len(pose_frames) / total if total > 0 else 0

        # ── D1: Eye-contact (iris-based) ──
        # Measured only on frames where the face is detected.
        # Face non-detection = missing data (patient may have turned head),
        # NOT evidence of gaze aversion.
        gaze_frames = [
            f for f in face_frames
            if f.get("eye_contact") and f["eye_contact"].get("on_target") is not None
        ]
        gaze_on_target_flags = [f["eye_contact"]["on_target"] for f in gaze_frames]

        iris_h_offsets = [
            f["eye_contact"]["iris_horizontal_offset"]
            for f in gaze_frames
            if f["eye_contact"].get("iris_horizontal_offset") is not None
        ]
        iris_v_offsets = [
            f["eye_contact"]["iris_vertical_offset"]
            for f in gaze_frames
            if f["eye_contact"].get("iris_vertical_offset") is not None
        ]

        d1 = {
            "gaze_on_target": _proportion_and_count(gaze_on_target_flags),
            "iris_horizontal_offset_distribution": _distribution_summary(iris_h_offsets),
            "iris_vertical_offset_distribution": _distribution_summary(iris_v_offsets),
            "data_availability_rate": round(face_rate, 3),
            "gaze_measurable_frames": len(gaze_frames),
            "total_frames": total,
            "reliability": _reliability_level(face_rate),
            "method_note": (
                "Gaze measured via iris landmark offset relative to eye corners "
                "(MediaPipe 478-point model). Offset 0.5 = iris centered = "
                "looking directly at camera (= at the patient). "
                "Thresholds: horizontal |offset - 0.5| < 0.175, vertical < 0.175. "
                "data_availability_rate = fraction of frames where face was "
                "detectable. Face non-detection is MISSING DATA — it may mean "
                "the patient turned their head, not that the clinician looked away. "
                "Only gaze_measurable_frames contribute to gaze_on_target."
            ),
        }

        # ── D2: Positioning ──
        # Primary metric: clinician eye_level_y relative to session horizon.
        #
        # The session horizon Y (estimated from room structural lines) is the
        # true camera eye-level in the scene. It corrects for the camera's
        # upward tilt — a face appearing centred in frame is NOT at eye level
        # if the camera is already tilted upward.
        #
        # Interpretation:
        #   eye_level_y < horizon_y  →  clinician ABOVE camera eye-level
        #                               (standing taller than patient) — UNFAVOURABLE
        #   eye_level_y ≈ horizon_y  →  clinician at patient eye level — GOOD
        #   eye_level_y > horizon_y  →  clinician BELOW camera eye-level
        #                               (crouched below patient) — rarely needed
        #
        # horizon_elevation = horizon_y - eye_level_y
        #   Positive → clinician is above horizon → standing over patient
        #   Zero     → at eye level
        #   Negative → clinician below horizon → crouched below patient

        horizon_y     = horizon_info.get("horizon_y") if horizon_info else None
        horizon_valid = horizon_info.get("horizon_valid", False) if horizon_info else False

        eye_level_ys = [
            f["positioning_and_posture"]["eye_level_y"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_y") is not None
        ]

        # Horizon-relative elevation: positive = clinician above horizon = standing over patient
        if horizon_valid and horizon_y is not None:
            horizon_elevations = [
                horizon_y - ey   # positive = above horizon = too tall
                for ey in eye_level_ys
            ]
            at_eye_level_flags = [
                abs(e) < HORIZON_AT_LEVEL_THRESHOLD
                for e in horizon_elevations
            ]
            above_horizon_flags = [e > HORIZON_AT_LEVEL_THRESHOLD for e in horizon_elevations]
        else:
            # Fallback: use session-median baseline deviation (less accurate)
            horizon_elevations  = []
            at_eye_level_flags  = []
            above_horizon_flags = []

        # Session-baseline deviation (always computed as secondary metric)
        eye_level_deviations = [
            f["positioning_and_posture"]["eye_level_baseline_deviation"]["eye_level_y_deviation"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_baseline_deviation")
            and f["positioning_and_posture"]["eye_level_baseline_deviation"].get("eye_level_y_deviation") is not None
        ]

        d2 = {
            # Primary: horizon-corrected metrics
            "horizon_y": round(horizon_y, 4) if horizon_valid and horizon_y is not None else None,
            "horizon_valid": horizon_valid,
            "horizon_elevation_distribution": _distribution_summary(horizon_elevations),
            "at_patient_eye_level_rate": _proportion_and_count(at_eye_level_flags),
            "above_patient_eye_level_rate": _proportion_and_count(above_horizon_flags),
            # Secondary: raw and baseline-relative
            "eye_level_y_distribution": _distribution_summary(eye_level_ys),
            "eye_level_y_deviation_from_session_baseline": _distribution_summary(eye_level_deviations),
            "session_baseline_eye_y": round(baseline.baseline_eye_y, 3) if baseline.eye_y_valid and baseline.baseline_eye_y is not None else None,
            # Horizon estimation quality
            "horizon_estimation": horizon_info if horizon_info else {"horizon_valid": False},
            "pose_detection_rate": round(pose_rate, 3),
            "reliability": horizon_info.get("reliability", "low") if horizon_valid else _reliability_level(pose_rate),
            "method_note": (
                "D2 positioning uses horizon-corrected eye level. "
                "Session horizon Y is estimated from structural horizontal lines "
                "(ceiling panels, door frames) in the room background via Hough "
                "line detection — this gives the true camera eye-level independent "
                "of camera tilt. "
                "horizon_elevation = horizon_y - clinician_eye_y: "
                "positive = clinician above camera eye-level (standing over patient), "
                "~0 = at patient eye level (favourable), "
                "negative = below patient eye level. "
                "at_patient_eye_level_rate = fraction of frames within ±4% of horizon. "
                + (
                    f"Horizon not reliably estimated — falling back to session-median "
                    f"baseline. Interpret D2 with caution."
                    if not horizon_valid else ""
                )
            ),
        }

        # ── D3: Posture ──
        arm_openness_vals = [
            f["positioning_and_posture"]["arm_openness"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("arm_openness") is not None
        ]
        landmark_confs = [
            f["positioning_and_posture"]["landmark_confidence"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]

        if baseline.valid and baseline.arm_valid:
            arm_deviations = [
                f["positioning_and_posture"]["baseline_deviation"]["arm_openness_deviation"]
                for f in pose_frames
                if f.get("positioning_and_posture")
                and f["positioning_and_posture"].get("baseline_deviation")
                and f["positioning_and_posture"]["baseline_deviation"].get("arm_openness_deviation") is not None
            ]
        else:
            arm_deviations = []

        d3 = {
            "arm_openness_distribution": _distribution_summary(arm_openness_vals),
            "baseline_arm_deviation": _distribution_summary(arm_deviations),
            "landmark_confidence_distribution": _distribution_summary(landmark_confs),
            "pose_detection_rate": round(pose_rate, 3),
            "reliability": _reliability_level(pose_rate),
            "method_note": (
                "Arm openness = wrist-to-wrist distance normalised by shoulder "
                "width. Values significantly below baseline suggest crossed arms. "
                "Camera roll from a seated patient is minimal; metric is reliable "
                "under normal conversational head movements."
            ),
        }

        # ── D4: Facial expressions ──
        smile_scores = [
            f["facial_expression"]["blendshape_smile"]
            for f in face_frames
            if f.get("facial_expression")
            and f["facial_expression"].get("blendshape_smile") is not None
        ]

        if baseline.valid and baseline.expression_valid:
            smile_deviations = [
                f["facial_expression"]["baseline_deviation"]["smile_deviation"]
                for f in face_frames
                if f.get("facial_expression") and f["facial_expression"].get("baseline_deviation")
            ]
            smile_z_scores = [
                f["facial_expression"]["baseline_deviation"]["smile_z_score"]
                for f in face_frames
                if f.get("facial_expression") and f["facial_expression"].get("baseline_deviation")
            ]
            positive_expression_frames = [z > SMILE_Z_THRESHOLD for z in smile_z_scores]
        else:
            smile_deviations, smile_z_scores, positive_expression_frames = [], [], []

        sources = [
            f["facial_expression"]["source"]
            for f in face_frames
            if f.get("facial_expression") and f["facial_expression"].get("source")
        ]

        d4 = {
            "smile_distribution": _distribution_summary(smile_scores),
            "baseline_smile_deviation": _distribution_summary(smile_deviations),
            "positive_expression_rate": _proportion_and_count(positive_expression_frames),
            "data_source_distribution": _value_distribution(sources),
            "face_detection_rate": round(face_rate, 3),
            "reliability": _reliability_level(face_rate),
            "method_note": (
                "Smile from MediaPipe blendshapes (mouthSmileLeft + mouthSmileRight / 2). "
                "Positive expression rate = frames where smile z-score > 1.0 above "
                "person's resting baseline."
            ),
        }

        # ── D5: Gestures & mannerisms ──
        # Head movement periodicity is NOT computed here. The camera is
        # head-mounted on the patient; yaw/pitch deltas between frames
        # reflect patient head movement (conversational nodding) and cannot
        # be separated from clinician movement. Only hand periodicity is
        # retained as a fidgeting indicator.
        gesture_frames = [f for f in frame_data if f.get("gestures")]
        hand_visible_rate = len(gesture_frames) / total if total > 0 else 0

        hand_spreads, hand_positions = [], []
        for f in gesture_frames:
            for h in f["gestures"].get("hands", []):
                hand_spreads.append(h["spread"])
                hand_positions.append(h["position"])

        effective_fps = video_fps

        left_hand_y_series, right_hand_y_series = [], []
        for f in gesture_frames:
            for h in f["gestures"].get("hands", []):
                if h["hand"] == "left":
                    left_hand_y_series.append(h["center_y"])
                elif h["hand"] == "right":
                    right_hand_y_series.append(h["center_y"])

        left_hand_periodicity  = _detect_periodicity(left_hand_y_series, effective_fps)
        right_hand_periodicity = _detect_periodicity(right_hand_y_series, effective_fps)

        if left_hand_periodicity["periodicity_strength"] >= right_hand_periodicity["periodicity_strength"]:
            hand_periodicity = {**left_hand_periodicity, "hand": "left"}
        else:
            hand_periodicity = {**right_hand_periodicity, "hand": "right"}

        d5 = {
            "hand_visibility_rate": round(hand_visible_rate, 3),
            "hand_spread_distribution": _distribution_summary(hand_spreads),
            "hand_position_distribution": _value_distribution(hand_positions),
            "hand_movement_periodicity": hand_periodicity,
            "reliability": _reliability_level(hand_visible_rate),
            "method_note": (
                "Fidgeting detected via autocorrelation periodicity analysis on "
                "hand Y-position time series. periodicity_strength > 0.3 indicates "
                "repetitive movement. Hand position relative to body landmarks. "
                "Head movement periodicity is intentionally not computed: the "
                "camera is head-mounted on the patient and patient conversational "
                "nodding is indistinguishable from clinician head movement."
            ),
        }

        # ── I: Professional behaviour demeanour ──
        d_i = {
            "gaze_on_target": d1["gaze_on_target"],
            "positive_expression_rate": d4["positive_expression_rate"],
            "arm_openness_distribution": d3["arm_openness_distribution"],
            "hand_movement_periodicity": d5["hand_movement_periodicity"],
            "overall_reliability": _reliability_level(min(face_rate, pose_rate)),
            "method_note": (
                "Video-observable demeanour cues only. LUCAS Item I is primarily "
                "determined by verbal behaviour — video metrics are supporting "
                "evidence and should not be sole basis for scoring."
            ),
        }

        return {
            "D1_eye_contact": d1,
            "D2_positioning": d2,
            "D3_posture": d3,
            "D4_facial_expressions": d4,
            "D5_gestures_and_mannerisms": d5,
            "I_professional_behaviour_demeanour": d_i,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Build LLM-optimised output JSON
    # ═══════════════════════════════════════════════════════════════════════
    def _build_llm_output(
        self,
        nvb_metrics: Dict,
        baseline: PersonBaseline,
        metadata: Dict,
    ) -> Dict[str, Any]:
        output = {
            "analysis_metadata": {
                "video_fps": metadata.get("video_fps"),
                "total_frames": metadata.get("total_frames"),
                "frames_analyzed": metadata.get("frames_analyzed"),
                "face_detection_rate": metadata.get("face_detection_rate"),
                "model": metadata.get("model"),
                "workers": metadata.get("workers"),
            },
            "person_baseline": baseline.to_dict(),
            "interpretation_guidance": {
                "camera_setup": (
                    "Egocentric (head-mounted) camera worn by the seated patient. "
                    "The camera optical axis approximates the patient's line of sight. "
                    "Camera pitch is approximately stable throughout the session."
                ),
                "note": (
                    "D1 uses iris landmark offsets (no baseline needed — self-referencing). "
                    "D2/D3 use person-relative baselines from first calibration window. "
                    "D4 uses person-relative smile baseline. "
                    "Distribution summaries are provided instead of binary classifications."
                ),
                "reliability_scale": {
                    "high":     "Detection rate >= 75% — metric is trustworthy",
                    "moderate": "Detection rate 40-75% — interpret with caution",
                    "low":      "Detection rate < 40% — metric is unreliable",
                },
                "d1_gaze_interpretation": {
                    "on_target": "iris_horizontal_offset ~0.5 AND iris_vertical_offset ~0.5 — clinician looking at patient",
                    "off_target": "iris offset > 0.175 from centre — clinician looking away (notes, monitor, floor etc.)",
                    "missing_data": "face_detected=False frames are excluded from gaze metric entirely",
                },
                "d2_positioning_interpretation": {
                    "horizon_y": "Estimated camera eye-level Y in normalised frame coords (0=top, 1=bottom), derived from room structural lines via Hough detection",
                    "horizon_elevation_positive": "clinician eye_level_y < horizon_y → clinician ABOVE camera eye-level → standing over patient (unfavourable)",
                    "horizon_elevation_zero": "clinician at patient eye level (favourable for paediatric consultation)",
                    "horizon_elevation_negative": "clinician eye_level_y > horizon_y → clinician below patient eye level (unusual)",
                    "at_patient_eye_level_rate": "proportion of frames where clinician is within ±8% of horizon (at eye level)",
                    "fallback_note": "If horizon_valid=False, fall back to session-baseline deviation — less accurate, interpret with caution",
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
        horizon_y: Optional[float] = None,
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

        draw_landmarks = cfg.get("annotated_draw_landmarks", True)
        feature_lookup = {f["frame_idx"]: f for f in frame_data}
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
                self._draw_overlay(frame, last_features, width, height, horizon_y=horizon_y)
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
                connection_color=(200, 200, 200),
                landmark_color=(200, 200, 200),
                img_w=img_w, img_h=img_h,
                connection_thickness=1,
                landmark_radius=2,
                min_idx=11,
            )

        face_landmarks = cached.get("face_landmarks")
        if face_landmarks is not None and len(face_landmarks) >= 468:
            VideoAnalysisStage._draw_landmarks_on_frame(
                frame, face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                connection_color=(160, 160, 160),
                landmark_color=None,
                img_w=img_w, img_h=img_h,
                connection_thickness=1,
            )

        for hand_key in ("left_hand_landmarks", "right_hand_landmarks"):
            hand_lms = cached.get(hand_key)
            if hand_lms and len(hand_lms) >= 21:
                VideoAnalysisStage._draw_landmarks_on_frame(
                    frame, hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    connection_color=(200, 200, 200),
                    landmark_color=(200, 200, 200),
                    img_w=img_w, img_h=img_h,
                    connection_thickness=1,
                    landmark_radius=2,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Overlay — LUCAS order, two-colour palette (white text, dim labels)
    # ═══════════════════════════════════════════════════════════════════════
    def _draw_overlay(self, frame, features, width, height, horizon_y: Optional[float] = None):
        """
        Left-panel overlay showing LUCAS items in order D1 → D2 → D3 → D4 → D5 → I.
        If horizon_y is provided, draws a horizontal dashed reference line across
        the full frame at the estimated camera eye-level.
        """
        import cv2

        FONT       = cv2.FONT_HERSHEY_DUPLEX
        SCALE_HDR  = 0.38
        SCALE_VAL  = 0.44
        SCALE_LBL  = 0.36
        THICK      = 1
        GAP        = 20
        GAP_S      = 14
        SEC        = 10

        DIM    = (110, 110, 110)
        BRIGHT = (220, 220, 220)
        ALERT  = (160, 160, 160)

        PANEL_W = 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (PANEL_W, height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # ── Draw horizon reference line across full frame ──
        if horizon_y is not None:
            hy_px = int(horizon_y * height)
            # Dashed line: draw segments every 20px, gap 10px
            dash_len, gap_len = 20, 10
            x = PANEL_W
            while x < width:
                x_end = min(x + dash_len, width)
                cv2.line(frame, (x, hy_px), (x_end, hy_px),
                         (160, 160, 160), 1, cv2.LINE_AA)
                x += dash_len + gap_len
            cv2.putText(frame, "cam level", (width - 110, hy_px - 4),
                        FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

        x = 12
        y = 30

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

        # ── D1 — Eye Contact (iris-based) ──
        header("D1  EYE CONTACT")
        ec = features.get("eye_contact")
        if ec:
            h_off = ec.get("iris_horizontal_offset")
            v_off = ec.get("iris_vertical_offset")
            on_tgt = ec.get("on_target", False)
            if h_off is not None:
                col = BRIGHT if on_tgt else ALERT
                v_txt = f"  V={v_off:.2f}" if v_off is not None else ""
                put(f"H={h_off:.2f}{v_txt}  {'on' if on_tgt else 'off'}", color=col, gap=GAP_S)
            else:
                put("Iris not detected", color=DIM, gap=GAP_S)
        else:
            put("No face detected", color=DIM, gap=GAP_S)
        spacer()

        # ── D2 — Positioning ──
        header("D2  POSITIONING")
        pp = features.get("positioning_and_posture")
        if pp and pp.get("eye_level_y") is not None:
            eye_y = pp["eye_level_y"]
            if horizon_y is not None:
                elev = horizon_y - eye_y   # positive = above horizon = standing over patient
                if abs(elev) < HORIZON_AT_LEVEL_THRESHOLD:
                    pos_label = "at level"
                    col = BRIGHT
                elif elev > 0:
                    pos_label = "above"
                    col = ALERT
                else:
                    pos_label = "below"
                    col = BRIGHT
                put(f"Eye Y {eye_y:.3f}  h={horizon_y:.3f}", color=DIM, gap=GAP_S)
                put(f"elev={elev:+.3f}  {pos_label}", color=col, gap=GAP_S)
            else:
                dev_info = pp.get("eye_level_baseline_deviation") or {}
                dev = dev_info.get("eye_level_y_deviation")
                dev_txt = f"  d={dev:+.3f}" if dev is not None else "  (no horizon)"
                put(f"Eye Y  {eye_y:.3f}{dev_txt}", gap=GAP_S)
        else:
            put("No pose detected", color=DIM, gap=GAP_S)
        spacer()

        # ── D3 — Posture ──
        header("D3  POSTURE")
        if pp:
            arm = pp.get("arm_openness")
            dev = (pp.get("baseline_deviation") or {})
            arm_dev = dev.get("arm_openness_deviation")
            if arm is not None:
                crossed = (arm_dev is not None and arm_dev < ARM_CROSSED_DEV) or \
                          (arm is not None and arm < ARM_CROSSED_ABS)
                col = ALERT if crossed else BRIGHT
                dev_txt = f"  d={arm_dev:+.2f}" if arm_dev is not None else ""
                put(f"Arm open  {arm:.2f}{dev_txt}", color=col, gap=GAP_S)
        else:
            put("No pose detected", color=DIM, gap=GAP_S)
        spacer()

        # ── D4 — Facial Expressions ──
        header("D4  EXPRESSION")
        fe = features.get("facial_expression")
        if fe and fe.get("blendshape_smile") is not None:
            smile = fe["blendshape_smile"]
            dev = (fe.get("baseline_deviation") or {})
            smile_z = dev.get("smile_z_score")
            z_txt = f"  z={smile_z:.2f}" if smile_z is not None else ""
            put(f"Smile  {smile:.3f}{z_txt}", gap=GAP_S)
            src = fe.get("source", "")
            if "fallback" in src:
                put("(landmark fallback)", color=DIM, scale=SCALE_LBL, gap=GAP_S)
        else:
            put("No face detected", color=DIM, gap=GAP_S)
        spacer()

        # ── D5 — Gestures & Mannerisms ──
        header("D5  GESTURES")
        gest = features.get("gestures")
        if gest and gest.get("num_hands_visible", 0) > 0:
            put(f"Hands  {gest['num_hands_visible']} visible", gap=GAP_S)
        else:
            put("No hands detected", color=DIM, gap=GAP_S)
        spacer()

        # ── I — Professional Behaviour summary ──
        header("Professional Behaviour Summary")
        if ec and ec.get("on_target") is not None:
            put(f"Gaze  {'on target' if ec['on_target'] else 'off target'}", gap=GAP_S)
        if fe and fe.get("baseline_deviation"):
            sz = (fe["baseline_deviation"] or {}).get("smile_z_score")
            if sz is not None:
                put(f"Expr  {'positive' if sz > SMILE_Z_THRESHOLD else 'neutral'}", gap=GAP_S)
        if pp and pp.get("arm_openness") is not None:
            arm = pp["arm_openness"]
            arm_d = (pp.get("baseline_deviation") or {}).get("arm_openness_deviation")
            crossed = (arm_d is not None and arm_d < ARM_CROSSED_DEV) or arm < ARM_CROSSED_ABS
            put(f"Arms  {'crossed' if crossed else 'open'}", gap=GAP_S)

        # ── Footer ──
        ts = features.get("timestamp_s", 0)
        cv2.putText(frame, f"{ts:.1f}s", (width - 70, height - 14),
                    FONT, 0.45, DIM, 1, cv2.LINE_AA)

        parts = []
        if features.get("face_detected"):
            parts.append("F")
        if features.get("pose_detected"):
            parts.append("P")
        if gest and gest.get("num_hands_visible", 0) > 0:
            parts.append("H")
        status = "+".join(parts) if parts else "none"
        cv2.putText(frame, f"Track: {status}", (x, height - 14),
                    FONT, 0.40, DIM, 1, cv2.LINE_AA)