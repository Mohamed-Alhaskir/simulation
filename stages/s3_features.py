"""
Stage 3: Feature Extraction
=============================
Extracts structured features from the diarized transcript:
  - Turn-taking patterns (who speaks when, how long)
  - Pause analysis (meaningful silences)
  - Interruption detection
  - Speaker ratios and conversation dynamics


Profile assembly (combining transcript, verbal features, and video features
into a single LLM-ready context) is handled by Stage 5 (LLMAnalysisStage),
which saves an assembled_context.json for auditability before building
the prompt.
"""

import json
from pathlib import Path

from stages.base import BaseStage
from utils.artifact_io import save_artifact


class FeatureExtractionStage(BaseStage):
    """Extract interaction features from the diarized transcript."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("features")
        output_dir = Path(ctx["output_base"]) / "03_features"
        output_dir.mkdir(parents=True, exist_ok=True)


        transcript = self._resolve_artifact(ctx["artifacts"]["transcript"])

        features = {}

        # ---- Verbal / interaction features ----
        verbal_cfg = cfg.get("verbal", {})
        features["verbal"] = self._extract_verbal_features(transcript, verbal_cfg)
        self.logger.info(
            f"Verbal features: {features['verbal']['summary']['total_turns']} turns, "
            f"{features['verbal']['summary']['total_speakers']} speakers"
        )

        # ---- Save features ----
        features_path = output_dir / "features.json"
        save_artifact(features, features_path, description="features", logger_instance=self.logger, default=str)

        ctx["artifacts"]["features"] = features
        ctx["artifacts"]["features_path"] = str(features_path)

        self.logger.info("Feature extraction complete")
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

        # Response latencies — collect into lists first, then aggregate,
        # to avoid mutating the dict while iterating over it.
        raw_latencies: dict[str, list[float]] = {}
        for i in range(1, len(turns)):
            if turns[i]["speaker"] != turns[i - 1]["speaker"]:
                gap = turns[i]["start"] - turns[i - 1]["end"]
                if 0 <= gap < 10:  # Ignore unreasonable gaps
                    raw_latencies.setdefault(turns[i]["speaker"], []).append(
                        round(gap, 2)
                    )

        response_latencies = {
            sp: {
                "mean_s": round(sum(lats) / len(lats), 2),
                "count": len(lats),
            }
            for sp, lats in raw_latencies.items()
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
