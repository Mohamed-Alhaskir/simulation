#!/usr/bin/env python3
"""
Paediatric Simulation AI Feedback Pipeline
===========================================
Frozen, version-locked pipeline for generating standardized feedback reports
from audiovisual recordings of paediatric simulation scenarios.

Stage order
-----------
  1. ingest         — validate video, extract audio, split quadrants
  2. asr            — transcription + speaker diarization
  3. features       — verbal interaction metrics + phase segmentation
  4. video_analysis — non-verbal behaviour (MediaPipe, LUCAS-aligned)
  5. analysis       — LLM assessment (assembles context from 2+3+4)
  6. report         — render standardized feedback report

Note: features (3) must run before video_analysis (4) because video
analysis uses ctx["artifacts"]["features"]["phases"] for per-phase NVB
summaries.

Usage
-----
    python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/
    python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch
    python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch --force
    python pipeline.py --config config/pipeline_config.yaml --freeze-manifest
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import numpy as np

# NumPy 2 compatibility shims for legacy libraries that reference removed aliases
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

from stages.s1_ingest import DataIngestionStage
from stages.s2_asr import ASRStage
from stages.s3_features import FeatureExtractionStage
from stages.s4_video_analysis import VideoAnalysisStage
from stages.s5_analysis import LLMAnalysisStage
from stages.s6_report import ReportGenerationStage
from utils.freeze import FreezeManifest
from utils.logging_setup import setup_logging
from utils.json_utils import JSONEncoder, sanitize_for_json

# ---------------------------------------------------------------------------
# Pipeline version — increment on ANY change to code, prompts, or models
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "0.3.0"

# ---------------------------------------------------------------------------
# Mapping from stage name → the output subdirectory that stage writes to.
# This is the single source of truth used by both the stage itself and the
# checkpoint logic, so they can never disagree.
# ---------------------------------------------------------------------------
STAGE_OUTPUT_DIRS: dict[str, str] = {
    "ingest":         "01_ingest",
    "asr":            "02_asr",
    "features":       "03_features",
    "video_analysis": "04_video_analysis",
    "analysis":       "05_analysis",
    "report":         "06_report",
}

# Stage execution order. features must precede video_analysis because
# video_analysis reads ctx["artifacts"]["features"]["phases"].
STAGE_ORDER = [
    "ingest",
    "asr",
    "features",
    "video_analysis",
    "analysis",
    "report",
]


class Pipeline:
    """End-to-end pipeline orchestrator."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("pipeline")

        self.manifest = FreezeManifest(
            pipeline_version=PIPELINE_VERSION,
            config=self.config,
        )

        self.stages = {
            "ingest":         DataIngestionStage(self.config),
            "asr":            ASRStage(self.config),
            "features":       FeatureExtractionStage(self.config),
            "video_analysis": VideoAnalysisStage(self.config),
            "analysis":       LLMAnalysisStage(self.config),
            "report":         ReportGenerationStage(self.config),
        }

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _checkpoint_path(output_base: Path, stage_name: str) -> Path:
        """Return the checkpoint file path for a given stage."""
        subdir = STAGE_OUTPUT_DIRS[stage_name]
        return output_base / subdir / ".stage_checkpoint.json"

    def _save_checkpoint(
        self, output_base: Path, stage_name: str, ctx: dict
    ) -> None:
        """Persist stage artifacts to a checkpoint file."""
        checkpoint_path = self._checkpoint_path(output_base, stage_name)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stage": stage_name,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "artifacts": sanitize_for_json(ctx["artifacts"]),
                        "timestamp": ctx["timestamps"].get(stage_name, {}),
                    },
                    f,
                    indent=2,
                    cls=JSONEncoder,
                )
        except Exception as e:
            self.logger.warning(
                f"Could not save checkpoint for '{stage_name}': {e}. "
                "Stage will re-run on next invocation."
            )

    def _load_checkpoint(
        self, output_base: Path, stage_name: str, ctx: dict
    ) -> bool:
        """
        Attempt to restore a stage from its checkpoint.

        Returns True if the checkpoint was loaded successfully, False otherwise.
        """
        checkpoint_path = self._checkpoint_path(output_base, stage_name)
        if not checkpoint_path.exists():
            return False
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ctx["artifacts"].update(data.get("artifacts", {}))
            ctx["timestamps"][stage_name] = data.get("timestamp", {})
            self.logger.info(
                f"    ↩ '{stage_name}' restored from checkpoint "
                f"(saved {data.get('saved_at', 'unknown')})"
            )
            return True
        except Exception as e:
            self.logger.warning(
                f"Checkpoint for '{stage_name}' is corrupt or unreadable "
                f"({e}). Re-running stage."
            )
            return False

    def _clear_checkpoints(self, output_base: Path) -> None:
        """Delete all stage checkpoints under output_base."""
        for stage_name in STAGE_ORDER:
            cp = self._checkpoint_path(output_base, stage_name)
            if cp.exists():
                cp.unlink()
                self.logger.info(f"Cleared checkpoint: {cp}")

    # ------------------------------------------------------------------
    # Single-session run
    # ------------------------------------------------------------------
    def run(
        self,
        input_path: str,
        session_id: str | None = None,
        force: bool = False,
    ) -> dict:
        """
        Execute all pipeline stages for a single simulation session.

        Parameters
        ----------
        input_path : str
            Path to session directory containing raw recordings.
        session_id : str | None
            Optional override; derived from directory name if None.
        force : bool
            If True, ignore existing checkpoints and re-run all stages.

        Returns
        -------
        dict
            Final pipeline context with all intermediate and final outputs.
        """
        input_dir = Path(input_path)
        if session_id is None:
            session_id = input_dir.name

        output_base = Path(self.config["paths"]["output_dir"]) / session_id
        output_base.mkdir(parents=True, exist_ok=True)

        if force:
            self.logger.info("--force: clearing existing checkpoints.")
            self._clear_checkpoints(output_base)

        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE START — session: {session_id}")
        self.logger.info(f"Pipeline version: {PIPELINE_VERSION}")
        self.logger.info(f"Freeze manifest hash: {self.manifest.digest()}")
        self.logger.info("=" * 70)

        ctx: dict = {
            "session_id": session_id,
            "input_path": str(input_dir),
            "output_base": str(output_base),
            "config": self.config,
            "manifest": self.manifest.to_dict(),
            "timestamps": {},
            "artifacts": {},
        }

        for stage_name in STAGE_ORDER:
            stage = self.stages[stage_name]
            self.logger.info(f"--- Stage: {stage_name} ---")

            # Try checkpoint restore first
            if self._load_checkpoint(output_base, stage_name, ctx):
                continue

            t0 = time.perf_counter()
            try:
                ctx = stage.run(ctx)
            except Exception:
                self.logger.exception(f"Stage '{stage_name}' failed.")
                ctx["timestamps"][stage_name] = {"status": "FAILED"}
                # Remove any partial checkpoint so a re-run starts fresh
                cp = self._checkpoint_path(output_base, stage_name)
                if cp.exists():
                    cp.unlink()
                raise

            elapsed = time.perf_counter() - t0
            ctx["timestamps"][stage_name] = {
                "status": "OK",
                "elapsed_s": round(elapsed, 2),
            }

            self._save_checkpoint(output_base, stage_name, ctx)
            self.logger.info(f"    ✓ {stage_name} completed in {elapsed:.2f}s")

        # Save pipeline run metadata
        meta_path = output_base / "pipeline_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "manifest_hash": self.manifest.digest(),
                    "timestamps": sanitize_for_json(ctx["timestamps"]),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
                cls=JSONEncoder,
            )

        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE COMPLETE — session: {session_id}")
        self.logger.info(
            f"Report: {ctx['artifacts'].get('report_path', 'N/A')}"
        )
        self.logger.info("=" * 70)

        return ctx

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------
    def run_batch(self, input_dir: str, force: bool = False) -> list[dict]:
        """
        Run pipeline on all session subdirectories.

        Parameters
        ----------
        input_dir : str
            Parent directory containing one subdirectory per session.
        force : bool
            If True, ignore existing checkpoints for all sessions.
        """
        root = Path(input_dir)
        sessions = sorted([d for d in root.iterdir() if d.is_dir()])
        self.logger.info(f"Batch mode: {len(sessions)} sessions found in {root}")

        results = []
        failed = []
        for session_dir in sessions:
            self.logger.info(f"\nProcessing session: {session_dir.name}")
            try:
                result = self.run(str(session_dir), force=force)
                results.append(result)
            except Exception as exc:
                self.logger.error(
                    f"Session '{session_dir.name}' failed: {exc}. Continuing."
                )
                failed.append(session_dir.name)
                results.append({
                    "session_id": session_dir.name,
                    "status": "FAILED",
                    "error": str(exc),
                })

        self.logger.info(
            f"\nBatch complete: {len(sessions) - len(failed)} succeeded, "
            f"{len(failed)} failed."
        )
        if failed:
            self.logger.warning(f"Failed sessions: {failed}")

        return results


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Paediatric Simulation AI Feedback Pipeline"
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to session directory (or parent dir in batch mode)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all subdirectories in --input as separate sessions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore existing stage checkpoints and re-run all stages. "
            "Use this after code or config changes to ensure a clean run."
        ),
    )
    parser.add_argument(
        "--freeze-manifest",
        action="store_true",
        help="Print the freeze manifest for the current config and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # --freeze-manifest is a read-only introspection tool — construct the
    # pipeline and print the manifest before any stage initialisation that
    # might fail due to missing model files.
    if args.freeze_manifest:
        config = Pipeline._load_config(args.config)
        manifest = FreezeManifest(
            pipeline_version=PIPELINE_VERSION,
            config=config,
        )
        print(json.dumps(manifest.to_dict(), indent=2))
        sys.exit(0)

    if not args.input:
        parser.error("--input is required unless using --freeze-manifest")

    pipe = Pipeline(args.config)

    if args.batch:
        pipe.run_batch(args.input, force=args.force)
    else:
        pipe.run(args.input, force=args.force)


if __name__ == "__main__":
    main()