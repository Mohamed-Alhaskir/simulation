#!/usr/bin/env python3
"""
Paediatric Simulation AI Feedback Pipeline
===========================================
Frozen, version-locked pipeline for generating standardized feedback reports
from audiovisual recordings of paediatric simulation scenarios.

Usage:
    python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/
    python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch
    python pipeline.py --freeze-manifest  # Print current freeze manifest and exit
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from stages.s1_ingest import DataIngestionStage
from stages.s2_asr import ASRStage
from stages.s3_features import FeatureExtractionStage
from stages.s3b_video_analysis import VideoAnalysisStage
#from stages.s3c_visual_diarization import VisualDiarizationStage
from stages.s4_analysis import LLMAnalysisStage
from stages.s5_report import ReportGenerationStage
from utils.freeze import FreezeManifest
from utils.logging_setup import setup_logging
from utils.json_utils import JSONEncoder, sanitize_for_json

# ---------------------------------------------------------------------------
# Pipeline version — increment on ANY change to code, prompts, or models
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "0.2.0-dev"

import numpy as np

# NumPy 2 compatibility shims for legacy libraries
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


class Pipeline:
    """End-to-end pipeline orchestrator."""

    STAGE_ORDER = [
        "ingest",
        "asr",
        #"features",
        "video_analysis",
        #"visual_diarization",   # refine speaker labels using mouth activity
        #"analysis",
        #"report",
    ]

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("pipeline")

        # Initialize freeze manifest
        self.manifest = FreezeManifest(
            pipeline_version=PIPELINE_VERSION,
            config=self.config,
        )

        # Initialize stages
        self.stages = {
            "ingest": DataIngestionStage(self.config),
            "asr": ASRStage(self.config),
            "video_analysis": VideoAnalysisStage(self.config),
            #"visual_diarization": VisualDiarizationStage(self.config),
            #"features": FeatureExtractionStage(self.config),
            #"analysis": LLMAnalysisStage(self.config),
            #"report": ReportGenerationStage(self.config),
        }

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, input_path: str, session_id: str | None = None) -> dict:
        """
        Execute all pipeline stages for a single simulation session.

        Parameters
        ----------
        input_path : str
            Path to session directory containing raw recordings.
        session_id : str | None
            Optional override; derived from directory name if None.

        Returns
        -------
        dict
            Final pipeline context with all intermediate and final outputs.
        """
        input_dir = Path(input_path)
        if session_id is None:
            session_id = input_dir.name

        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE START — session: {session_id}")
        self.logger.info(f"Freeze manifest hash: {self.manifest.digest()}")
        self.logger.info("=" * 70)

        # Pipeline context — passed through all stages
        ctx = {
            "session_id": session_id,
            "input_path": str(input_dir),
            "output_base": str(Path(self.config["paths"]["output_dir"]) / session_id),
            "config": self.config,
            "manifest": self.manifest.to_dict(),
            "timestamps": {},
            "artifacts": {},
        }

        # Ensure output directory
        Path(ctx["output_base"]).mkdir(parents=True, exist_ok=True)

        for i, stage_name in enumerate(self.STAGE_ORDER, start=1):
            stage = self.stages[stage_name]
            self.logger.info(f"--- Stage: {stage_name} ---")
            
            # Check if stage already completed
            stage_output_dir = Path(ctx["output_base"]) / f"0{i}_{stage_name}"
            stage_checkpoint = stage_output_dir / ".stage_checkpoint.json"
            
            if stage_checkpoint.exists():
                self.logger.info(f"Stage '{stage_name}' already completed, loading cached results...")
                try:
                    with open(stage_checkpoint, "r") as f:
                        checkpoint_data = json.load(f)
                    ctx["artifacts"].update(checkpoint_data.get("artifacts", {}))
                    ctx["timestamps"][stage_name] = checkpoint_data.get("timestamp", {})
                    self.logger.info(f"    ✓ {stage_name} restored from cache")
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to restore checkpoint: {e}. Re-running stage...")
            
            t0 = time.perf_counter()

            try:
                ctx = stage.run(ctx)
            except Exception:
                self.logger.exception(f"Stage '{stage_name}' failed.")
                ctx["timestamps"][stage_name] = {"status": "FAILED"}
                raise

            elapsed = time.perf_counter() - t0
            ctx["timestamps"][stage_name] = {
                "status": "OK",
                "elapsed_s": round(elapsed, 2),
            }
            
            # Save checkpoint for this stage
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            with open(stage_checkpoint, "w") as f:
                json.dump({
                    "artifacts": sanitize_for_json(ctx["artifacts"]),
                    "timestamp": ctx["timestamps"][stage_name],
                }, f, indent=2, cls=JSONEncoder)
            
            self.logger.info(
                f"    ✓ {stage_name} completed in {elapsed:.2f}s"
            )

        # Save pipeline run metadata
        meta_path = Path(ctx["output_base"]) / "pipeline_meta.json"
        with open(meta_path, "w") as f:
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
        self.logger.info(f"Report: {ctx['artifacts'].get('report_path', 'N/A')}")
        self.logger.info("=" * 70)

        return ctx

    def run_batch(self, input_dir: str) -> list[dict]:
        """Run pipeline on all session subdirectories."""
        root = Path(input_dir)
        sessions = sorted([d for d in root.iterdir() if d.is_dir()])
        self.logger.info(f"Batch mode: found {len(sessions)} sessions")

        results = []
        for session_dir in sessions:
            try:
                result = self.run(str(session_dir))
                results.append(result)
            except Exception:
                self.logger.error(f"Session {session_dir.name} failed, continuing...")
                results.append({"session_id": session_dir.name, "status": "FAILED"})

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
        help="Process all subdirectories in --input",
    )
    parser.add_argument(
        "--freeze-manifest",
        action="store_true",
        help="Print freeze manifest and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    pipe = Pipeline(args.config)

    if args.freeze_manifest:
        print(json.dumps(pipe.manifest.to_dict(), indent=2))
        sys.exit(0)

    if not args.input:
        parser.error("--input is required unless using --freeze-manifest")

    if args.batch:
        pipe.run_batch(args.input)
    else:
        pipe.run(args.input)


if __name__ == "__main__":
    main()