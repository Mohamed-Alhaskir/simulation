"""
Stage 3b: Video-Based Non-Verbal Behaviour Analysis (LUCAS-Aligned)
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

Output JSON is structured for downstream LLM scoring against the LUCAS
rubric (Competent / Borderline / Unacceptable per sub-item).
"""

import base64, json, math, os, urllib.request
from pathlib import Path
from typing import Optional
import numpy as np
from stages.base import BaseStage

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ---------------------------------------------------------------------------
# MediaPipe model downloads
# ---------------------------------------------------------------------------
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
_MODELS = {
    "face": (
        f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker.task",
    ),
    "pose": (
        f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "pose_landmarker_heavy.task",
    ),
    "hand": (
        f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task",
    ),
}
_MODEL_DIR = Path.home() / ".cache" / "mediapipe" / "models"


def _ensure_model(key):
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
# Note: Landmark connection definitions are sourced directly from the
# MediaPipe Tasks API (PoseLandmarksConnections, HandLandmarksConnections,
# FaceLandmarksConnections) rather than hardcoded lists.
# ---------------------------------------------------------------------------


class VideoAnalysisStage(BaseStage):
    """LUCAS-aligned NVB video analysis stage."""

    # ── Face landmark indices ──
    NOSE_TIP = 1
    LEFT_EYE_INNER = 133; RIGHT_EYE_INNER = 362
    LEFT_EYE_OUTER = 33;  RIGHT_EYE_OUTER = 263
    CHIN = 152; FOREHEAD = 10
    LEFT_MOUTH = 61; RIGHT_MOUTH = 291
    UPPER_LIP = 13; LOWER_LIP = 14
    LEFT_EYE_TOP = 159; LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386; RIGHT_EYE_BOTTOM = 374
    LEFT_BROW = 70; RIGHT_BROW = 300

    # ── Pose landmark indices ──
    POSE_NOSE = 0
    POSE_LEFT_SHOULDER = 11; POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_ELBOW = 13;    POSE_RIGHT_ELBOW = 14
    POSE_LEFT_WRIST = 15;    POSE_RIGHT_WRIST = 16
    POSE_LEFT_HIP = 23;      POSE_RIGHT_HIP = 24

    # ═══════════════════════════════════════════════════════════════════════
    # run()
    # ═══════════════════════════════════════════════════════════════════════
    def run(self, ctx):
        cfg = self._get_stage_config("video_analysis")
        output_dir = Path(ctx["output_base"]) / "03b_video_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not cfg.get("enabled", False):
            self.logger.info("Video analysis disabled in config, skipping.")
            return ctx

        video_path = self._resolve_video_source(ctx, cfg)
        if video_path is None:
            self.logger.warning("No video source available, skipping video analysis.")
            return ctx

        self.logger.info(f"Video source: {video_path}")

        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            self.logger.error("opencv-python and/or mediapipe not installed.")
            raise

        for key in ("face", "pose", "hand"):
            _ensure_model(key)

        # ── Retrieve conversation phase info (if available) ──
        features = ctx["artifacts"].get("features", {})
        phases = features.get("phases", [])
        sample_fps = cfg.get("sample_fps", 2)

        # ── Run frame-by-frame analysis ──
        results = self._analyze_video(video_path, sample_fps, cfg)
        self.logger.info(
            f"Analyzed {results['metadata']['frames_analyzed']} frames "
            f"({results['metadata']['faces_detected']} with face detected)"
        )

        # ── Compute LUCAS-aligned NVB metrics ──
        nvb_metrics = self._compute_lucas_nvb_metrics(results["frame_data"])
        phase_summaries = self._compute_phase_summaries(results["frame_data"], phases)

        # ── Generate annotated video (optional) ──
        annotated_video_path = None
        if cfg.get("generate_annotated_video", True):
            annotated_video_path = str(output_dir / "annotated_video.mp4")
            self._generate_annotated_video(
                video_path=video_path,
                frame_data=results["frame_data"],
                cached_landmarks=results.get("cached_landmarks", {}),
                output_path=annotated_video_path,
                cfg=cfg,
            )

        # ── Build results-only output JSON ──
        lucas_nvb_output = self._build_llm_output(
            nvb_metrics=nvb_metrics,
            phase_summaries=phase_summaries,
        )

        features_path = output_dir / "video_features.json"
        with open(features_path, "w") as f:
            json.dump(lucas_nvb_output, f, indent=2, ensure_ascii=False, default=str)

        # ── Update pipeline context ──
        llm_profile = ctx["artifacts"].get("llm_profile", {})
        llm_profile["lucas_nvb"] = lucas_nvb_output
        ctx["artifacts"]["llm_profile"] = llm_profile
        ctx["artifacts"]["video_features"] = lucas_nvb_output
        ctx["artifacts"]["video_features_path"] = str(features_path)
        if annotated_video_path:
            ctx["artifacts"]["annotated_video_path"] = annotated_video_path

        return ctx

    # ═══════════════════════════════════════════════════════════════════════
    # Video source resolution
    # ═══════════════════════════════════════════════════════════════════════
    def _resolve_video_source(self, ctx, cfg):
        preferred_quadrant = cfg.get("preferred_quadrant", "bottom_right")
        quadrants = ctx["artifacts"].get("inventory", {}).get("quadrants", {})
        if preferred_quadrant in quadrants:
            return quadrants[preferred_quadrant]
        composite = ctx["artifacts"].get("composite_video")
        if composite and Path(composite).exists():
            return composite
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Parallel video analysis (structure preserved)
    # ═══════════════════════════════════════════════════════════════════════
    def _analyze_video(self, video_path, sample_fps, cfg):
        import cv2
        from multiprocessing import get_context
        from functools import partial

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_interval = 1
        num_workers = 40
        segment_size = max(1, total_frames // num_workers)

        self.logger.info(
            f"[Parallel] {video_fps:.1f} FPS | {total_frames} frames | "
            f"interval={frame_interval} | workers={num_workers}"
        )

        segments = []
        for i in range(num_workers):
            start = i * segment_size
            end = (i + 1) * segment_size if i < num_workers - 1 else total_frames
            segments.append((video_path, start, end, frame_interval, cfg))

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

        # Compute inter-frame head movement deltas
        for i in range(1, len(frame_data)):
            frame_data[i]["movement_delta"] = self._compute_movement_delta(
                frame_data[i - 1], frame_data[i]
            )

        return {
            "metadata": {
                "video_path": video_path,
                "video_fps": video_fps,
                "total_frames": total_frames,
                "sample_fps": sample_fps,
                "frames_analyzed": frames_analyzed,
                "faces_detected": faces_detected,
                "face_detection_rate": round(faces_detected / frames_analyzed, 3)
                if frames_analyzed > 0
                else 0,
                "model": "mediapipe_tasks_face+pose+hand_parallel",
                "workers": num_workers,
            },
            "frame_data": frame_data,
            "cached_landmarks": cached_landmarks,
        }

    def _process_video_segment(self, args):
        import os, cv2, mediapipe as mp

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

        video_path, start_frame, end_frame, frame_interval, cfg = args

        face_model_path = _ensure_model("face")
        pose_model_path = _ensure_model("pose")
        hand_model_path = _ensure_model("hand")

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        det_conf = cfg.get("detection_confidence", 0.5)
        trk_conf = cfg.get("tracking_confidence", 0.5)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=trk_conf,
            min_tracking_confidence=trk_conf,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=trk_conf,
            min_tracking_confidence=trk_conf,
        )
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO, num_hands=2,
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

        face_lm = FaceLandmarker.create_from_options(face_options)
        pose_lm = PoseLandmarker.create_from_options(pose_options)
        hand_lm = HandLandmarker.create_from_options(hand_options)

        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    timestamp_ms = int((frame_idx / video_fps) * 1000)
                    timestamp_s = frame_idx / video_fps
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    face_result = face_lm.detect_for_video(mp_image, timestamp_ms)
                    pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
                    hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)

                    h, w, _ = frame.shape
                    frame_features = {
                        "timestamp_s": round(timestamp_s, 2),
                        "frame_idx": frame_idx,
                    }

                    face_landmarks = (
                        face_result.face_landmarks[0]
                        if face_result.face_landmarks else None
                    )
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
                    }

                    # ── D1: Eye-contact (from head pose / gaze) ──
                    if face_landmarks and len(face_landmarks) > 0:
                        faces_detected += 1
                        frame_features["face_detected"] = True
                        frame_features["eye_contact"] = self._estimate_eye_contact(
                            face_landmarks, w, h
                        )
                        # ── D4: Facial expressions ──
                        frame_features["facial_expression"] = (
                            self._extract_facial_expression(face_landmarks, w, h)
                        )
                    else:
                        frame_features["face_detected"] = False
                        frame_features["eye_contact"] = None
                        frame_features["facial_expression"] = None

                    # ── D2: Positioning & D3: Posture ──
                    if pose_landmarks and len(pose_landmarks) > 0:
                        frame_features["pose_detected"] = True
                        frame_features["positioning_and_posture"] = (
                            self._analyze_positioning_posture(
                                pose_landmarks,
                                face_landmarks=face_landmarks,
                                img_w=w, img_h=h,
                            )
                        )
                    else:
                        frame_features["pose_detected"] = False
                        frame_features["positioning_and_posture"] = None

                    # ── D5: Gestures & mannerisms ──
                    has_left = left_hand_lms and len(left_hand_lms) > 0
                    has_right = right_hand_lms and len(right_hand_lms) > 0
                    if has_left or has_right:
                        frame_features["gestures"] = self._analyze_gestures(
                            left_hand_lms if has_left else None,
                            right_hand_lms if has_right else None,
                            pose_landmarks, w, h,
                        )
                    else:
                        frame_features["gestures"] = None

                    frame_data.append(frame_features)
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
    # Inter-frame movement delta (for fidget / mannerism detection)
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_movement_delta(self, prev_frame, curr_frame):
        delta = {
            "head_yaw_delta": None,
            "head_pitch_delta": None,
        }
        prev_ec = prev_frame.get("eye_contact")
        curr_ec = curr_frame.get("eye_contact")
        if prev_ec and curr_ec:
            delta["head_yaw_delta"] = abs(
                curr_ec.get("yaw", 0) - prev_ec.get("yaw", 0)
            )
            delta["head_pitch_delta"] = abs(
                curr_ec.get("pitch", 0) - prev_ec.get("pitch", 0)
            )
        return delta

    # ═══════════════════════════════════════════════════════════════════════
    # D1 — Eye-contact
    # ═══════════════════════════════════════════════════════════════════════
    def _estimate_eye_contact(self, landmarks, img_w, img_h):
        """Head-pose-based gaze estimation → LUCAS D1 eye-contact."""

        def lm_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

        nose = lm_pt(self.NOSE_TIP)
        chin = lm_pt(self.CHIN)
        forehead = lm_pt(self.FOREHEAD)
        left_eye = lm_pt(self.LEFT_EYE_OUTER)
        right_eye = lm_pt(self.RIGHT_EYE_OUTER)

        eye_center = (left_eye + right_eye) / 2
        eye_width = np.linalg.norm(right_eye - left_eye)
        yaw = (
            math.degrees(math.atan2(nose[0] - eye_center[0], eye_width))
            if eye_width > 0 else 0.0
        )
        face_height = np.linalg.norm(forehead - chin)
        pitch = (
            math.degrees(math.atan2(nose[1] - eye_center[1], face_height))
            if face_height > 0 else 0.0
        )

        gaze = "forward"
        if abs(yaw) > 20:
            gaze = "left" if yaw < 0 else "right"
        elif abs(pitch) > 15:
            gaze = "down" if pitch > 0 else "up"

        looking_at_patient = abs(yaw) < 25 and abs(pitch) < 20

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "gaze_direction": gaze,
            "looking_at_patient": looking_at_patient,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # D4 — Facial expressions
    # ═══════════════════════════════════════════════════════════════════════
    def _extract_facial_expression(self, landmarks, img_w, img_h):
        """Extract expression features → LUCAS D4 facial expressions."""

        def lm_2d(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        upper_lip = lm_2d(self.UPPER_LIP)
        lower_lip = lm_2d(self.LOWER_LIP)
        left_mouth = lm_2d(self.LEFT_MOUTH)
        right_mouth = lm_2d(self.RIGHT_MOUTH)

        mouth_height = np.linalg.norm(lower_lip - upper_lip)
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        # Brow height (concern / surprise indicator)
        nose_tip = lm_2d(self.NOSE_TIP)
        chin = lm_2d(self.CHIN)
        face_height = np.linalg.norm(chin - nose_tip)

        left_brow = lm_2d(self.LEFT_BROW)
        right_brow = lm_2d(self.RIGHT_BROW)
        left_eye_center = lm_2d(self.LEFT_EYE_INNER)
        right_eye_center = lm_2d(self.RIGHT_EYE_INNER)
        avg_brow_height = (
            (left_eye_center[1] - left_brow[1])
            + (right_eye_center[1] - right_brow[1])
        ) / 2
        normalized_brow = avg_brow_height / face_height if face_height > 0 else 0

        # Smile score
        mouth_center = (left_mouth + right_mouth) / 2
        smile_score = (
            (mouth_center[1] - left_mouth[1])
            + (mouth_center[1] - right_mouth[1])
        ) / 2
        normalized_smile = smile_score / face_height if face_height > 0 else 0

        expression = self._classify_expression(
            mouth_aspect_ratio, normalized_brow, normalized_smile
        )

        return {
            "smile_score": round(float(normalized_smile), 3),
            "brow_height": round(float(normalized_brow), 3),
            "mouth_open": mouth_aspect_ratio > 0.15,
            "expression": expression,
        }

    @staticmethod
    def _classify_expression(mouth_ratio, brow_height, smile):
        if smile > 0.02:
            return "positive"
        elif brow_height > 0.35:
            return "concerned"
        elif mouth_ratio > 0.3:
            return "speaking"
        else:
            return "neutral"

    # ═══════════════════════════════════════════════════════════════════════
    # D2 — Positioning  &  D3 — Posture
    # ═══════════════════════════════════════════════════════════════════════
    def _analyze_positioning_posture(self, pose_landmarks, face_landmarks=None,
                                      img_w=None, img_h=None):
        """Shoulder alignment, lean, arm position, height-level → LUCAS D2 + D3."""

        def plm(idx):
            lm = pose_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z]), getattr(lm, "visibility", 1.0)

        left_shoulder, ls_vis = plm(self.POSE_LEFT_SHOULDER)
        right_shoulder, rs_vis = plm(self.POSE_RIGHT_SHOULDER)
        left_hip, lh_vis = plm(self.POSE_LEFT_HIP)
        right_hip, rh_vis = plm(self.POSE_RIGHT_HIP)
        left_wrist, lw_vis = plm(self.POSE_LEFT_WRIST)
        right_wrist, rw_vis = plm(self.POSE_RIGHT_WRIST)

        # Shoulder tilt (D2 — positioning symmetry)
        shoulder_diff_y = right_shoulder[1] - left_shoulder[1]
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_tilt = (
            math.degrees(math.atan2(shoulder_diff_y, shoulder_width))
            if shoulder_width > 0 else 0
        )

        # Torso lean (D2 — forward / back positioning, D3 — posture)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_vector = shoulder_center - hip_center
        torso_lean = math.degrees(
            math.atan2(torso_vector[2], abs(torso_vector[1]))
        )

        # Arm openness (D3 — posture, crossed arms indicator)
        wrist_distance = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
        arm_openness = (
            wrist_distance / shoulder_width if shoulder_width > 0 else 1.0
        )

        # Labels
        if abs(shoulder_tilt) > 10:
            posture_label = "asymmetric"
        elif torso_lean > 10:
            posture_label = "leaning_forward"
        elif torso_lean < -5:
            posture_label = "leaning_back"
        else:
            posture_label = "upright"

        if arm_openness < 0.3:
            arm_label = "crossed"
        elif arm_openness > 1.5:
            arm_label = "wide"
        else:
            arm_label = "natural"

        avg_vis = np.mean([ls_vis, rs_vis, lh_vis, rh_vis])

        # ── D2: Same-height positioning ──
        # In this setup the camera represents the patient's viewpoint.
        # The student's eye-level Y position (normalised 0=top, 1=bottom)
        # indicates whether they are seated at the same height as the patient.
        #   - eye_y ≈ 0.30–0.55  → same height  (eyes in mid-frame)
        #   - eye_y < 0.30       → student higher than patient
        #   - eye_y > 0.55       → student lower  than patient
        eye_level_y = None
        height_level = "unknown"
        if face_landmarks and len(face_landmarks) > max(
            self.LEFT_EYE_INNER, self.RIGHT_EYE_INNER
        ):
            left_eye_y = face_landmarks[self.LEFT_EYE_INNER].y
            right_eye_y = face_landmarks[self.RIGHT_EYE_INNER].y
            eye_level_y = round(float((left_eye_y + right_eye_y) / 2), 3)
            if 0.30 <= eye_level_y <= 0.55:
                height_level = "same_height"
            elif eye_level_y < 0.30:
                height_level = "above_patient"
            else:
                height_level = "below_patient"

        return {
            "shoulder_tilt_deg": round(shoulder_tilt, 1),
            "torso_lean_deg": round(torso_lean, 1),
            "posture_label": posture_label,
            "arm_openness": round(float(arm_openness), 2),
            "arm_position": arm_label,
            "eye_level_y": eye_level_y,
            "height_level": height_level,
            "confidence": round(float(avg_vis), 2),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # D5 — Gestures & mannerisms
    # ═══════════════════════════════════════════════════════════════════════
    def _analyze_gestures(self, left_hand_lms, right_hand_lms,
                          pose_landmarks, img_w, img_h):
        """Hand visibility, spread, position → LUCAS D5 gestures."""
        hands_info = []
        for label, hand_lms in [("left", left_hand_lms),
                                 ("right", right_hand_lms)]:
            if hand_lms is None or len(hand_lms) == 0:
                continue
            wrist = hand_lms[0]
            middle_tip = hand_lms[12]
            cy = np.mean([lm.y for lm in hand_lms])
            spread = np.linalg.norm(
                np.array([middle_tip.x - wrist.x, middle_tip.y - wrist.y])
            )

            # Position relative to body
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
                "position": position,
            })

        return {
            "num_hands_visible": len(hands_info),
            "hands": hands_info,
            "gesturing": any(h["spread"] > 0.1 for h in hands_info),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate LUCAS NVB metrics
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_lucas_nvb_metrics(self, frame_data):
        """Aggregate per-frame features into LUCAS D and I sub-item metrics."""
        if not frame_data:
            return {}

        total = len(frame_data)
        face_frames = [f for f in frame_data if f.get("face_detected")]

        # ── D1: Eye-contact ──
        looking_frames = [
            f for f in face_frames
            if f.get("eye_contact", {}).get("looking_at_patient", False)
        ]
        eye_contact_rate = (
            len(looking_frames) / len(face_frames) if face_frames else 0
        )

        yaws = [
            f["eye_contact"]["yaw"]
            for f in face_frames if f.get("eye_contact")
        ]
        pitches = [
            f["eye_contact"]["pitch"]
            for f in face_frames if f.get("eye_contact")
        ]
        gaze_stability_yaw_std = float(np.std(yaws)) if yaws else 0
        gaze_stability_pitch_std = float(np.std(pitches)) if pitches else 0

        # ── D2: Positioning ──
        pose_frames = [f for f in frame_data if f.get("pose_detected")]
        lean_angles = [
            f["positioning_and_posture"]["torso_lean_deg"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]
        shoulder_tilts = [
            f["positioning_and_posture"]["shoulder_tilt_deg"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]
        avg_lean = float(np.mean(lean_angles)) if lean_angles else 0
        avg_shoulder_tilt = float(np.mean(np.abs(shoulder_tilts))) if shoulder_tilts else 0

        # Same-height positioning (student at patient eye-level)
        height_levels = [
            f["positioning_and_posture"]["height_level"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("height_level") != "unknown"
        ]
        height_counts = {}
        for hl in height_levels:
            height_counts[hl] = height_counts.get(hl, 0) + 1
        height_distribution = (
            {k: round(v / len(height_levels), 3) for k, v in height_counts.items()}
            if height_levels else {}
        )
        same_height_rate = height_distribution.get("same_height", 0)

        eye_level_ys = [
            f["positioning_and_posture"]["eye_level_y"]
            for f in pose_frames
            if f.get("positioning_and_posture")
            and f["positioning_and_posture"].get("eye_level_y") is not None
        ]
        avg_eye_level_y = round(float(np.mean(eye_level_ys)), 3) if eye_level_ys else None

        # ── D3: Posture ──
        posture_labels = [
            f["positioning_and_posture"]["posture_label"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]
        posture_counts = {}
        for pl in posture_labels:
            posture_counts[pl] = posture_counts.get(pl, 0) + 1
        posture_distribution = (
            {k: round(v / len(posture_labels), 3) for k, v in posture_counts.items()}
            if posture_labels else {}
        )

        arm_positions = [
            f["positioning_and_posture"]["arm_position"]
            for f in pose_frames if f.get("positioning_and_posture")
        ]
        arm_counts = {}
        for ap in arm_positions:
            arm_counts[ap] = arm_counts.get(ap, 0) + 1
        arm_distribution = (
            {k: round(v / len(arm_positions), 3) for k, v in arm_counts.items()}
            if arm_positions else {}
        )

        # ── D4: Facial expressions ──
        expressions = [
            f["facial_expression"]["expression"]
            for f in face_frames if f.get("facial_expression")
        ]
        expr_counts = {}
        for expr in expressions:
            expr_counts[expr] = expr_counts.get(expr, 0) + 1
        expression_distribution = (
            {k: round(v / len(expressions), 3) for k, v in expr_counts.items()}
            if expressions else {}
        )
        smile_scores = [
            f["facial_expression"]["smile_score"]
            for f in face_frames if f.get("facial_expression")
        ]
        avg_smile = float(np.mean(smile_scores)) if smile_scores else 0

        facilitative_rate = (
            expression_distribution.get("positive", 0)
            + expression_distribution.get("neutral", 0)
        )

        # ── D5: Gestures & mannerisms ──
        gesture_frames = [
            f for f in frame_data
            if f.get("gestures") and f["gestures"].get("gesturing", False)
        ]
        gesture_rate = len(gesture_frames) / total if total > 0 else 0

        head_deltas = [
            f["movement_delta"]["head_yaw_delta"]
            for f in frame_data
            if f.get("movement_delta", {}).get("head_yaw_delta") is not None
        ]
        avg_head_movement = float(np.mean(head_deltas)) if head_deltas else 0
        fidget_score = float(np.std(head_deltas)) if head_deltas else 0

        # ── I: Professional behaviour (demeanour proxy) ──
        crossed_rate = arm_distribution.get("crossed", 0)

        return {
            "D1_eye_contact": {
                "eye_contact_rate": round(eye_contact_rate, 3),
                "gaze_stability": {
                    "yaw_std": round(gaze_stability_yaw_std, 1),
                    "pitch_std": round(gaze_stability_pitch_std, 1),
                },
                "description": (
                    f"Student maintained eye-contact with the patient in "
                    f"{eye_contact_rate:.0%} of detected frames. "
                    f"Gaze stability: yaw SD={gaze_stability_yaw_std:.1f}°, "
                    f"pitch SD={gaze_stability_pitch_std:.1f}°."
                ),
            },
            "D2_positioning": {
                "avg_forward_lean_deg": round(avg_lean, 1),
                "avg_shoulder_tilt_deg": round(avg_shoulder_tilt, 1),
                "same_height_rate": round(same_height_rate, 3),
                "height_distribution": height_distribution,
                "avg_eye_level_y": avg_eye_level_y,
                "description": (
                    f"Average forward lean: {avg_lean:.1f}° "
                    f"(positive = toward patient). "
                    f"Mean absolute shoulder tilt: {avg_shoulder_tilt:.1f}°. "
                    f"Same height as patient: {same_height_rate:.0%} of frames "
                    f"(height breakdown: {height_distribution}). "
                    f"Average eye-level Y position: {avg_eye_level_y} "
                    f"(0.0 = top of frame, 1.0 = bottom; "
                    f"0.30–0.55 = same height as patient)."
                ),
            },
            "D3_posture": {
                "posture_distribution": posture_distribution,
                "arm_position_distribution": arm_distribution,
                "description": (
                    f"Posture breakdown: {posture_distribution}. "
                    f"Arm position breakdown: {arm_distribution}."
                ),
            },
            "D4_facial_expressions": {
                "expression_distribution": expression_distribution,
                "avg_smile_score": round(avg_smile, 3),
                "facilitative_expression_rate": round(facilitative_rate, 3),
                "description": (
                    f"Expression breakdown: {expression_distribution}. "
                    f"Facilitative (positive + neutral): {facilitative_rate:.0%}. "
                    f"Average smile score: {avg_smile:.3f}."
                ),
            },
            "D5_gestures_and_mannerisms": {
                "gesture_rate": round(gesture_rate, 3),
                "avg_head_movement_deg": round(avg_head_movement, 2),
                "fidget_score": round(fidget_score, 2),
                "description": (
                    f"Active gesturing in {gesture_rate:.0%} of frames. "
                    f"Average inter-frame head movement: {avg_head_movement:.2f}°. "
                    f"Fidget score (head movement variability): {fidget_score:.2f}."
                ),
            },
            "I_professional_behaviour_demeanour": {
                "crossed_arms_rate": round(crossed_rate, 3),
                "fidget_score": round(fidget_score, 2),
                "eye_contact_rate": round(eye_contact_rate, 3),
                "facilitative_expression_rate": round(facilitative_rate, 3),
                "description": (
                    f"Crossed-arms rate: {crossed_rate:.0%}. "
                    f"Fidget score: {fidget_score:.2f}. "
                    f"Eye-contact: {eye_contact_rate:.0%}. "
                    f"Facilitative expressions: {facilitative_rate:.0%}. "
                    f"These video-observable cues contribute to the overall "
                    f"impression of courteous, kind, and professional demeanour."
                ),
            },
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Phase-level summaries (reuses the same LUCAS metrics per phase)
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_phase_summaries(self, frame_data, phases):
        if not phases or not frame_data:
            return []
        summaries = []
        for phase in phases:
            p_start = phase.get("start_s", 0)
            p_end = phase.get("end_s", 0)
            phase_frames = [
                f for f in frame_data if p_start <= f["timestamp_s"] <= p_end
            ]
            if not phase_frames:
                summaries.append({
                    "phase": phase.get("phase", "unknown"),
                    "start_s": p_start, "end_s": p_end,
                    "frames_analyzed": 0,
                })
                continue

            # Re-compute LUCAS metrics for this phase window
            summary = self._compute_lucas_nvb_metrics(phase_frames)
            summary["phase"] = phase.get("phase", "unknown")
            summary["start_s"] = p_start
            summary["end_s"] = p_end
            summary["frames_analyzed"] = len(phase_frames)
            summaries.append(summary)
        return summaries

    # ═══════════════════════════════════════════════════════════════════════
    # Build LLM-optimised output JSON
    # ═══════════════════════════════════════════════════════════════════════
    def _build_llm_output(self, nvb_metrics, phase_summaries):
        """
        Returns only the NVB results — no metadata, prompts, or rubric.
        The downstream LLM prompt supplies its own LUCAS context.
        """
        output = dict(nvb_metrics)
        if phase_summaries:
            output["phase_summaries"] = phase_summaries
        return output

    # ═══════════════════════════════════════════════════════════════════════
    # Annotated video generation (structure preserved from original)
    # ═══════════════════════════════════════════════════════════════════════
    def _generate_annotated_video(self, video_path, frame_data,
                                  cached_landmarks, output_path, cfg):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video for annotation: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*cfg.get("annotated_codec", "mp4v"))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            self.logger.error(f"Cannot open video writer for: {output_path}")
            cap.release()
            return

        draw_landmarks = cfg.get("annotated_draw_landmarks", True)
        feature_lookup = {f["frame_idx"]: f for f in frame_data}
        frame_idx, last_features = 0, None

        self.logger.info(
            f"Rendering annotated video: {width}x{height} @ {fps} FPS ..."
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if draw_landmarks and frame_idx in cached_landmarks:
                self._draw_landmarks_cv2(
                    frame, cached_landmarks[frame_idx], width, height
                )
            if frame_idx in feature_lookup:
                last_features = feature_lookup[frame_idx]
            if last_features:
                self._draw_overlay(frame, last_features, width, height)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        self.logger.info(
            f"Annotated video complete: {frame_idx} frames -> {output_path}"
        )

    @staticmethod
    def _draw_landmarks_on_frame(frame, landmarks, connections,
                                 connection_color, landmark_color,
                                 img_w, img_h, connection_thickness=2,
                                 landmark_radius=2, min_idx=0):
        """
        Unified landmark drawing using MediaPipe Tasks API Connection objects.

        Parameters
        ----------
        frame           : BGR numpy image
        landmarks       : list of mediapipe NormalizedLandmark (x, y in [0,1])
        connections     : list of Connection(start, end) from Tasks API
                          (e.g. PoseLandmarksConnections.POSE_LANDMARKS)
        connection_color: BGR tuple for lines
        landmark_color  : BGR tuple for points (None to skip points)
        img_w, img_h    : pixel dimensions of frame
        connection_thickness : line thickness
        landmark_radius : circle radius for landmark dots
        min_idx         : only draw landmarks with index >= min_idx
        """
        import cv2

        if landmarks is None or len(landmarks) == 0:
            return

        n = len(landmarks)

        # Draw connections
        for conn in connections:
            a, b = conn.start, conn.end
            if a < n and b < n:
                pt_a = (int(landmarks[a].x * img_w),
                        int(landmarks[a].y * img_h))
                pt_b = (int(landmarks[b].x * img_w),
                        int(landmarks[b].y * img_h))
                cv2.line(frame, pt_a, pt_b, connection_color,
                         connection_thickness, cv2.LINE_AA)

        # Draw landmark points
        if landmark_color is not None:
            for idx in range(min_idx, n):
                pt = (int(landmarks[idx].x * img_w),
                      int(landmarks[idx].y * img_h))
                cv2.circle(frame, pt, landmark_radius, landmark_color,
                           -1, cv2.LINE_AA)

    @staticmethod
    def _draw_landmarks_cv2(frame, cached, img_w, img_h):
        """
        Draw all landmarks using the unified drawing method with
        MediaPipe Tasks API native connection sets.
        """
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarksConnections,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarksConnections,
        )
        from mediapipe.tasks.python.vision.hand_landmarker import (
            HandLandmarksConnections,
        )

        # ── Face ──
        face_landmarks = cached.get("face_landmarks")
        if face_landmarks is not None and len(face_landmarks) >= 468:
            # Tesselation (subtle mesh)
            VideoAnalysisStage._draw_landmarks_on_frame(
                frame, face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                connection_color=(200, 200, 200),
                landmark_color=None,  # skip dots for tesselation
                img_w=img_w, img_h=img_h,
                connection_thickness=1,
            )
            # Contours (eyes, brows, lips, oval — brighter)
            #VideoAnalysisStage._draw_landmarks_on_frame(
            #    frame, face_landmarks,
            #    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            #    connection_color=(255, 255, 255),
            #    landmark_color=None,
            #    img_w=img_w, img_h=img_h,
            #    connection_thickness=1,
            #)

        # ── Pose ──
        pose_lms = cached.get("pose_landmarks")
        if pose_lms and len(pose_lms) >= 33:
            VideoAnalysisStage._draw_landmarks_on_frame(
                frame, pose_lms,
                connections=PoseLandmarksConnections.POSE_LANDMARKS,
                connection_color=(255, 255, 255),
                landmark_color=(255, 255, 255),
                img_w=img_w, img_h=img_h,
                connection_thickness=2,
                landmark_radius=3,
                min_idx=11,  # skip face landmarks (0-10) drawn by face mesh
            )

        # ── Hands ──
        for hand_key in ("left_hand_landmarks", "right_hand_landmarks"):
            hand_lms = cached.get(hand_key)
            if hand_lms and len(hand_lms) >= 21:
                VideoAnalysisStage._draw_landmarks_on_frame(
                    frame, hand_lms,
                    connections=HandLandmarksConnections.HAND_CONNECTIONS,
                    connection_color=(255, 255, 255),
                    landmark_color=(255, 255, 255),
                    img_w=img_w, img_h=img_h,
                    connection_thickness=2,
                    landmark_radius=2,
                )

    def _draw_overlay(self, frame, features, width, height):
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        x_left = 20
        y = 35
        line_gap = 26

        ts = features.get("timestamp_s", 0)
        cv2.putText(
            frame, f"{ts:.1f}s", (width - 100, height - 20),
            font, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # D1: Eye-contact
        ec = features.get("eye_contact")
        if ec:
            gaze_color = (0, 255, 0) if ec.get("looking_at_patient") else (0, 0, 255)
            cv2.putText(
                frame,
                f"Gaze: {ec.get('gaze_direction', 'unknown')} | "
                f"Yaw: {ec['yaw']} Pitch: {ec['pitch']}",
                (x_left, y), font, font_scale, gaze_color, thickness, cv2.LINE_AA,
            )
            y += line_gap

        # D4: Expression
        fe = features.get("facial_expression")
        if fe:
            cv2.putText(
                frame,
                f"Expression: {fe.get('expression', 'neutral')} | "
                f"Smile: {fe.get('smile_score', 0):.2f}",
                (x_left, y), font, font_scale, (255, 200, 0), thickness, cv2.LINE_AA,
            )
            y += line_gap

        # D2+D3: Posture & positioning
        pp = features.get("positioning_and_posture")
        if pp:
            height_lbl = pp.get("height_level", "?")
            height_color = (0, 255, 0) if height_lbl == "same_height" else (0, 165, 255)
            cv2.putText(
                frame,
                f"Posture: {pp.get('posture_label', '?')} | "
                f"Lean: {pp.get('torso_lean_deg', 0)}\xb0 | "
                f"Arms: {pp.get('arm_position', '?')}",
                (x_left, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
            y += line_gap
            cv2.putText(
                frame,
                f"Height: {height_lbl}",
                (x_left, y), font, font_scale, height_color, thickness, cv2.LINE_AA,
            )
            y += line_gap

        # D5: Gestures
        gest = features.get("gestures")
        if gest and gest.get("gesturing"):
            cv2.putText(
                frame, "Gesturing",
                (x_left, y), font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA,
            )
            y += line_gap

        # Engagement gauge
        engagement = self._frame_engagement_proxy(features)
        self._draw_engagement_gauge(
            frame, x=20, y=height - 90, w=320, h=70, value=engagement,
            col_g=(0, 200, 0), col_y=(0, 200, 200), col_r=(0, 0, 200),
            col_bg=(40, 40, 40), col_txt=(255, 255, 255),
            font=font, font_sz=0.5,
        )

    @staticmethod
    def _frame_engagement_proxy(feat):
        """Quick per-frame engagement estimate for the overlay gauge."""
        score = 0.0
        ec = feat.get("eye_contact")
        if ec and ec.get("looking_at_patient"):
            score += 0.40
        fe = feat.get("facial_expression")
        if fe:
            if fe.get("expression") == "positive":
                score += 0.20
            elif fe.get("expression") == "concerned":
                score += 0.10
        pp = feat.get("positioning_and_posture")
        if pp:
            lean = pp.get("torso_lean_deg", 0)
            if lean > 2:
                score += 0.20
            elif lean > 0:
                score += 0.10
            if pp.get("arm_position") == "natural":
                score += 0.05
        gest = feat.get("gestures")
        if gest and gest.get("gesturing"):
            score += 0.15
        return min(score, 1.0)

    @staticmethod
    def _draw_engagement_gauge(frame, x, y, w, h, value,
                               col_g, col_y, col_r, col_bg, col_txt,
                               font, font_sz):
        import cv2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), col_bg, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        bar_w = int((w - 8) * max(0.0, min(value, 1.0)))
        bar_x, bar_y, bar_h = x + 4, y + 18, h - 22
        bar_col = col_g if value >= 0.6 else (col_y if value >= 0.3 else col_r)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w - 8, bar_y + bar_h), col_txt, 1)
        cv2.putText(
            frame, f"Engagement: {value:.0%}",
            (x + 4, y + 14), font, font_sz, col_txt, 1, cv2.LINE_AA,
        )