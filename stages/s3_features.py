"""
Stage 3: Multimodal Feature Extraction
=======================================
Extracts structured features from:
- Diarized transcript → turn-taking, pauses, interruptions, speech ratios
- Physiological data   → HRV summary, EDA peaks (stress indicators)
- Eye-tracking data    → fixation patterns, gaze metrics

These features are assembled into a structured profile that feeds the LLM.
"""

import json
from pathlib import Path

import numpy as np

from stages.base import BaseStage


class FeatureExtractionStage(BaseStage):
    """Extract multimodal features for LLM analysis."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("features")
        output_dir = Path(ctx["output_base"]) / "03_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript = ctx["artifacts"]["transcript"]
        features = {}

        # ---- Verbal / Interaction features ----
        if cfg.get("verbal", {}).get("compute_turn_taking", True):
            features["verbal"] = self._extract_verbal_features(transcript, cfg["verbal"])
            self.logger.info(
                f"Verbal features: {len(features['verbal']['turns'])} turns, "
                f"{features['verbal']['summary']['total_speakers']} speakers"
            )

        # ---- Physiological features ----
        if cfg.get("physio", {}).get("enabled", False):
            inventory = ctx["artifacts"].get("inventory", {})
            physio_files = inventory.get("physio", [])
            if physio_files:
                features["physio"] = self._extract_physio_features(
                    physio_files, cfg["physio"]
                )
                self.logger.info("Physiological features extracted")
            else:
                self.logger.info("No physiological data available, skipping")
                features["physio"] = None

        # ---- Eye-tracking features ----
        if cfg.get("eyetracking", {}).get("enabled", False):
            inventory = ctx["artifacts"].get("inventory", {})
            et_files = inventory.get("eyetracking", [])
            if et_files:
                features["eyetracking"] = self._extract_eyetracking_features(
                    et_files, cfg["eyetracking"]
                )
                self.logger.info("Eye-tracking features extracted")
            else:
                self.logger.info("No eye-tracking data available, skipping")
                features["eyetracking"] = None

        # Save features
        features_path = output_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2, ensure_ascii=False, default=str)

        # Build the structured profile for the LLM
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
    # Verbal / Interaction features
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

        # Speaker statistics
        speakers = {}
        for turn in turns:
            sp = turn["speaker"]
            if sp not in speakers:
                speakers[sp] = {
                    "turn_count": 0,
                    "total_duration_s": 0,
                    "word_count": 0,
                }
            speakers[sp]["turn_count"] += 1
            speakers[sp]["total_duration_s"] += turn["duration_s"]
            speakers[sp]["word_count"] += len(turn["text"].split())

        total_duration = sum(s["total_duration_s"] for s in speakers.values())
        for sp in speakers.values():
            sp["total_duration_s"] = round(sp["total_duration_s"], 2)
            sp["speaking_ratio"] = (
                round(sp["total_duration_s"] / total_duration, 3)
                if total_duration > 0
                else 0
            )

        # Pauses between turns
        pauses = []
        for i in range(1, len(turns)):
            gap = turns[i]["start"] - turns[i - 1]["end"]
            if gap >= pause_threshold:
                pauses.append({
                    "after_turn": i - 1,
                    "before_turn": i,
                    "duration_s": round(gap, 2),
                    "between_speakers": (
                        turns[i - 1]["speaker"],
                        turns[i]["speaker"],
                    ),
                })

        # Interruptions (overlap: next turn starts before previous ends)
        interruptions = []
        if cfg.get("compute_interruptions", True):
            for i in range(1, len(turns)):
                overlap = turns[i - 1]["end"] - turns[i]["start"]
                if overlap > 0.3 and turns[i]["speaker"] != turns[i - 1]["speaker"]:
                    interruptions.append({
                        "turn": i,
                        "interrupter": turns[i]["speaker"],
                        "interrupted": turns[i - 1]["speaker"],
                        "overlap_s": round(overlap, 2),
                    })

        summary = {
            "total_speakers": len(speakers),
            "total_turns": len(turns),
            "total_duration_s": round(total_duration, 2),
            "meaningful_pauses": len(pauses),
            "interruptions": len(interruptions),
            "speakers": speakers,
        }

        return {
            "turns": turns,
            "pauses": pauses,
            "interruptions": interruptions,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Physiological features
    # ------------------------------------------------------------------
    def _extract_physio_features(self, physio_files: list[dict], cfg: dict) -> dict:
        """
        Extract heart rate variability and electrodermal activity summaries.
        Expects CSV files with columns: timestamp, hr, eda (at minimum).
        """
        import csv

        results = {}
        for finfo in physio_files:
            fpath = Path(finfo.get("processed_path", finfo["path"]))
            if fpath.suffix != ".csv":
                continue

            try:
                with open(fpath) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if not rows:
                    continue

                # Heart rate
                hr_values = [
                    float(r["hr"]) for r in rows
                    if "hr" in r and r["hr"]
                ]
                if hr_values:
                    arr = np.array(hr_values)
                    results["heart_rate"] = {
                        "mean": round(float(np.mean(arr)), 1),
                        "std": round(float(np.std(arr)), 1),
                        "min": round(float(np.min(arr)), 1),
                        "max": round(float(np.max(arr)), 1),
                    }

                # EDA
                eda_values = [
                    float(r["eda"]) for r in rows
                    if "eda" in r and r["eda"]
                ]
                if eda_values:
                    arr = np.array(eda_values)
                    results["eda"] = {
                        "mean": round(float(np.mean(arr)), 3),
                        "std": round(float(np.std(arr)), 3),
                        "max": round(float(np.max(arr)), 3),
                    }

            except (KeyError, ValueError) as e:
                self.logger.warning(f"Could not parse physio file {fpath}: {e}")

        return results if results else None

    # ------------------------------------------------------------------
    # Eye-tracking features
    # ------------------------------------------------------------------
    def _extract_eyetracking_features(
        self, et_files: list[dict], cfg: dict
    ) -> dict:
        """
        Extract gaze metrics from eye-tracking data.
        Expects CSV with columns: timestamp, x, y, fixation (bool/int).
        """
        import csv

        fixation_threshold_ms = cfg.get("fixation_threshold_ms", 100)
        results = {}

        for finfo in et_files:
            fpath = Path(finfo.get("processed_path", finfo["path"]))
            if fpath.suffix != ".csv":
                continue

            try:
                with open(fpath) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if not rows:
                    continue

                total_samples = len(rows)
                fixations = [r for r in rows if str(r.get("fixation", "0")) == "1"]

                results["gaze"] = {
                    "total_samples": total_samples,
                    "fixation_count": len(fixations),
                    "fixation_ratio": round(len(fixations) / total_samples, 3)
                    if total_samples > 0
                    else 0,
                }

            except (KeyError, ValueError) as e:
                self.logger.warning(f"Could not parse eye-tracking file {fpath}: {e}")

        return results if results else None

    # ------------------------------------------------------------------
    # Build LLM profile
    # ------------------------------------------------------------------
    def _build_llm_profile(self, features: dict, transcript: list[dict]) -> dict:
        """
        Assemble a structured profile document that will be inserted into
        the LLM prompt for analysis.
        """
        profile = {
            "transcript_summary": {
                "total_segments": len(transcript),
                "full_text_by_speaker": self._group_text_by_speaker(transcript),
            },
        }

        verbal = features.get("verbal")
        if verbal:
            profile["interaction"] = {
                "speaker_stats": verbal["summary"]["speakers"],
                "total_turns": verbal["summary"]["total_turns"],
                "meaningful_pauses": verbal["summary"]["meaningful_pauses"],
                "interruptions": verbal["summary"]["interruptions"],
                "pause_details": verbal["pauses"][:10],  # Limit for context window
                "interruption_details": verbal["interruptions"][:10],
            }

        physio = features.get("physio")
        if physio:
            profile["physiological"] = physio

        et = features.get("eyetracking")
        if et:
            profile["eyetracking"] = et

        # Include the full diarized transcript
        profile["diarized_transcript"] = [
            {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in transcript
        ]

        return profile

    @staticmethod
    def _group_text_by_speaker(transcript: list[dict]) -> dict:
        """Group all utterances by speaker."""
        grouped = {}
        for seg in transcript:
            sp = seg["speaker"]
            if sp not in grouped:
                grouped[sp] = []
            grouped[sp].append(seg["text"])
        return {sp: " ".join(texts) for sp, texts in grouped.items()}
