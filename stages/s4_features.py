"""
Stage 3: Feature Extraction
=============================
Extracts structured features from the diarized transcript:
  - Turn-taking patterns (who speaks when, how long)
  - Pause analysis (meaningful silences)
  - Interruption detection
  - Speaker ratios and conversation dynamics
  - Conversation phase segmentation

Optionally extracts vitals from the patient monitor quadrant via OCR
(top-right quadrant of the composite video).

Note: No separate physio or eye-tracking CSV files exist in this setup.
      Eye-tracking is only visible in the bottom-right video quadrant.
"""

import json
from pathlib import Path

from stages.base import BaseStage


class FeatureExtractionStage(BaseStage):
    """Extract interaction features from the diarized transcript."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("features")
        output_dir = Path(ctx["output_base"]) / "03_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript = ctx["artifacts"]["transcript"]
        features = {}

        # ---- Verbal / interaction features ----
        verbal_cfg = cfg.get("verbal", {})
        features["verbal"] = self._extract_verbal_features(transcript, verbal_cfg)
        self.logger.info(
            f"Verbal features: {features['verbal']['summary']['total_turns']} turns, "
            f"{features['verbal']['summary']['total_speakers']} speakers"
        )

        # ---- Conversation phase segmentation ----
        features["phases"] = self._segment_conversation_phases(
            features["verbal"]["turns"]
        )
        self.logger.info(
            f"Conversation phases: {len(features['phases'])} phases detected"
        )

        # ---- Patient monitor OCR (optional) ----
        monitor_cfg = cfg.get("monitor_ocr", {})
        if monitor_cfg.get("enabled", False):
            quadrants = ctx["artifacts"].get("inventory", {}).get("quadrants", {})
            monitor_video = quadrants.get("top_right")
            if monitor_video:
                features["vitals"] = self._extract_vitals_ocr(monitor_video)
                if features["vitals"]:
                    self.logger.info("Patient monitor vitals extracted via OCR")
            else:
                self.logger.info("No monitor quadrant available, skipping OCR")
                features["vitals"] = None
        else:
            features["vitals"] = None

        # ---- Save features ----
        features_path = output_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2, ensure_ascii=False, default=str)

        # ---- Build LLM profile ----
        llm_profile = self._build_llm_profile(features, transcript)
        profile_path = output_dir / "llm_profile.json"
        with open(profile_path, "w") as f:
            json.dump(llm_profile, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["features"] = features
        ctx["artifacts"]["features_path"] = str(features_path)
        ctx["artifacts"]["llm_profile"] = llm_profile
        ctx["artifacts"]["llm_profile_path"] = str(profile_path)

        return ctx

    # ------------------------------------------------------------------
    # Verbal / interaction features
    # ------------------------------------------------------------------
    def _extract_verbal_features(self, transcript: list[dict], cfg: dict) -> dict:
        """Compute interaction metrics from the diarized transcript."""
        pause_threshold = cfg.get("pause_threshold_s", 2.0)

        # Group consecutive segments by speaker into "turns"
        turns = []
        current_turn = None

        for seg in transcript:
            if current_turn is None or seg["speaker"] != current_turn["speaker"]:
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "segment_count": 1,
                }
            else:
                current_turn["end"] = seg["end"]
                current_turn["text"] += " " + seg["text"]
                current_turn["segment_count"] += 1

        if current_turn:
            turns.append(current_turn)

        # Compute turn durations
        for turn in turns:
            turn["duration_s"] = round(turn["end"] - turn["start"], 2)
            turn["word_count"] = len(turn["text"].split())

        # Speaker statistics
        speakers = {}
        for turn in turns:
            sp = turn["speaker"]
            if sp not in speakers:
                speakers[sp] = {
                    "turn_count": 0,
                    "total_duration_s": 0,
                    "word_count": 0,
                    "avg_turn_duration_s": 0,
                }
            speakers[sp]["turn_count"] += 1
            speakers[sp]["total_duration_s"] += turn["duration_s"]
            speakers[sp]["word_count"] += turn["word_count"]

        total_duration = sum(s["total_duration_s"] for s in speakers.values())
        for sp_name, sp_data in speakers.items():
            sp_data["total_duration_s"] = round(sp_data["total_duration_s"], 2)
            sp_data["speaking_ratio"] = (
                round(sp_data["total_duration_s"] / total_duration, 3)
                if total_duration > 0
                else 0
            )
            sp_data["avg_turn_duration_s"] = (
                round(sp_data["total_duration_s"] / sp_data["turn_count"], 2)
                if sp_data["turn_count"] > 0
                else 0
            )

        # Pauses between turns
        pauses = []
        for i in range(1, len(turns)):
            gap = turns[i]["start"] - turns[i - 1]["end"]
            if gap >= pause_threshold:
                pauses.append({
                    "after_turn_index": i - 1,
                    "duration_s": round(gap, 2),
                    "between": [turns[i - 1]["speaker"], turns[i]["speaker"]],
                    "timestamp_s": round(turns[i - 1]["end"], 2),
                })

        # Interruptions (next turn starts before previous ends)
        interruptions = []
        if cfg.get("compute_interruptions", True):
            for i in range(1, len(turns)):
                overlap = turns[i - 1]["end"] - turns[i]["start"]
                if overlap > 0.3 and turns[i]["speaker"] != turns[i - 1]["speaker"]:
                    interruptions.append({
                        "turn_index": i,
                        "interrupter": turns[i]["speaker"],
                        "interrupted": turns[i - 1]["speaker"],
                        "overlap_s": round(overlap, 2),
                        "timestamp_s": round(turns[i]["start"], 2),
                    })

        # Response latencies (how quickly does each speaker respond)
        response_latencies = {}
        for i in range(1, len(turns)):
            if turns[i]["speaker"] != turns[i - 1]["speaker"]:
                gap = turns[i]["start"] - turns[i - 1]["end"]
                if 0 <= gap < 10:  # Ignore unreasonable gaps
                    sp = turns[i]["speaker"]
                    if sp not in response_latencies:
                        response_latencies[sp] = []
                    response_latencies[sp].append(round(gap, 2))

        for sp, latencies in response_latencies.items():
            response_latencies[sp] = {
                "mean_s": round(sum(latencies) / len(latencies), 2),
                "count": len(latencies),
            }

        summary = {
            "total_speakers": len(speakers),
            "total_turns": len(turns),
            "total_duration_s": round(total_duration, 2),
            "meaningful_pauses": len(pauses),
            "interruptions": len(interruptions),
            "speakers": speakers,
            "response_latencies": response_latencies,
        }

        return {
            "turns": turns,
            "pauses": pauses,
            "interruptions": interruptions,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Conversation phase segmentation
    # ------------------------------------------------------------------
    def _segment_conversation_phases(self, turns: list[dict]) -> list[dict]:
        """
        Simple heuristic segmentation of conversation into phases:
        opening, history-taking, examination/discussion, closing.

        Based on temporal position within the conversation.
        """
        if not turns:
            return []

        total_duration = turns[-1]["end"] - turns[0]["start"]
        if total_duration <= 0:
            return []

        phases = []
        for turn in turns:
            relative_pos = (turn["start"] - turns[0]["start"]) / total_duration

            if relative_pos < 0.10:
                phase = "opening"
            elif relative_pos < 0.70:
                phase = "main_consultation"
            elif relative_pos < 0.90:
                phase = "summary_and_plan"
            else:
                phase = "closing"

            if not phases or phases[-1]["phase"] != phase:
                phases.append({
                    "phase": phase,
                    "start_s": round(turn["start"], 2),
                    "end_s": round(turn["end"], 2),
                    "turn_count": 1,
                })
            else:
                phases[-1]["end_s"] = round(turn["end"], 2)
                phases[-1]["turn_count"] += 1

        # Calculate durations
        for phase in phases:
            phase["duration_s"] = round(phase["end_s"] - phase["start_s"], 2)

        return phases

    # ------------------------------------------------------------------
    # Patient monitor OCR (placeholder)
    # ------------------------------------------------------------------
    def _extract_vitals_ocr(self, monitor_video_path: str) -> dict | None:
        """
        Extract vital signs from the patient monitor quadrant video.

        This is a placeholder — in production, you would use:
        - Tesseract OCR on sampled frames
        - Or a specialized medical display reader

        For now, returns None. The vital signs visible in your video are:
        HR, BP (NIBP), SpO2, Temp, displayed on a Philips monitor.
        """
        self.logger.info(
            "Monitor OCR: placeholder — implement with Tesseract or "
            "specialized reader for Philips IntelliVue displays"
        )
        # TODO: Implement OCR extraction
        # Sample frames at regular intervals (e.g., every 30s)
        # Run OCR on the vital sign regions
        # Parse HR, BP, SpO2, Temp values over time
        return None

    # ------------------------------------------------------------------
    # Build LLM profile
    # ------------------------------------------------------------------
    def _build_llm_profile(
        self, features: dict, transcript: list[dict]
    ) -> dict:
        """
        Assemble the structured profile document for the LLM prompt.
        """

        profile = {
            # Transcript summary
            "transcript_summary": {
                "total_segments": len(transcript),
                "total_duration_s": features["verbal"]["summary"]["total_duration_s"],
                "speakers": features["verbal"]["summary"]["speakers"],
            },
            # Interaction metrics
            "interaction": {
                "total_turns": features["verbal"]["summary"]["total_turns"],
                "meaningful_pauses": features["verbal"]["summary"]["meaningful_pauses"],
                "interruptions": features["verbal"]["summary"]["interruptions"],
                "response_latencies": features["verbal"]["summary"]["response_latencies"],
                "pause_details": features["verbal"]["pauses"][:10],
                "interruption_details": features["verbal"]["interruptions"][:10],
            },
            # Conversation phases
            "conversation_phases": features.get("phases", []),
            # Full diarized transcript
            "diarized_transcript": [
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in transcript
            ],
        }

        # Vitals from monitor (if extracted)
        if features.get("vitals"):
            profile["patient_vitals"] = features["vitals"]

        return profile
