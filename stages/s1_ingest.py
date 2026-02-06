"""
Stage 1: Data Ingestion & Validation
=====================================
- Scans session directory for recordings and supplementary data
- Validates file formats, durations, and completeness
- Copies/links accepted files to processed directory
- Flags sessions that fail quality gates (CONSORT-AI 5(iii))
"""

import json
import shutil
import subprocess
from pathlib import Path

from stages.base import BaseStage


class DataIngestionStage(BaseStage):
    """Ingest and validate raw simulation session data."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("ingest")
        input_dir = Path(ctx["input_path"])
        output_dir = Path(ctx["output_base"]) / "01_ingested"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Scanning input directory: {input_dir}")

        # Discover files
        inventory = {
            "video": self._find_files(input_dir, cfg.get("accepted_video_formats", [])),
            "audio": self._find_files(input_dir, cfg.get("accepted_audio_formats", [])),
            "physio": [],
            "eyetracking": [],
        }

        if cfg.get("physio", {}).get("enabled"):
            inventory["physio"] = self._find_files(
                input_dir, cfg["physio"].get("accepted_formats", [])
            )

        if cfg.get("eyetracking", {}).get("enabled"):
            inventory["eyetracking"] = self._find_files(
                input_dir, cfg["eyetracking"].get("accepted_formats", [])
            )

        # Log inventory
        for modality, files in inventory.items():
            self.logger.info(f"  {modality}: {len(files)} file(s)")

        # Validate: must have at least video OR audio
        if not inventory["video"] and not inventory["audio"]:
            raise ValueError(
                f"Session {ctx['session_id']}: No valid video or audio recordings found. "
                "Excluding per CONSORT-AI 4a(ii) / 5(iii)."
            )

        # Validate durations
        validated = {}
        for modality in ["video", "audio"]:
            validated[modality] = []
            for fpath in inventory[modality]:
                duration = self._get_media_duration(fpath)
                if duration is None:
                    self.logger.warning(f"  Cannot read duration: {fpath}, skipping")
                    continue
                if duration < cfg.get("min_duration_s", 30):
                    self.logger.warning(
                        f"  Too short ({duration:.1f}s): {fpath}, skipping"
                    )
                    continue
                if duration > cfg.get("max_duration_s", 3600):
                    self.logger.warning(
                        f"  Too long ({duration:.1f}s): {fpath}, skipping"
                    )
                    continue
                validated[modality].append({"path": str(fpath), "duration_s": duration})

        if not validated["video"] and not validated["audio"]:
            raise ValueError(
                f"Session {ctx['session_id']}: All recordings failed quality checks."
            )

        # Copy validated files to processed directory
        for modality, files in validated.items():
            for finfo in files:
                src = Path(finfo["path"])
                dst = output_dir / modality / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                finfo["processed_path"] = str(dst)

        # Include supplementary data as-is
        for modality in ["physio", "eyetracking"]:
            validated[modality] = []
            for fpath in inventory[modality]:
                dst = output_dir / modality / fpath.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, dst)
                validated[modality].append({
                    "path": str(fpath),
                    "processed_path": str(dst),
                })

        # Save inventory
        inventory_path = output_dir / "inventory.json"
        with open(inventory_path, "w") as f:
            json.dump(validated, f, indent=2)

        # Update context
        ctx["artifacts"]["inventory"] = validated
        ctx["artifacts"]["inventory_path"] = str(inventory_path)

        # Determine primary audio source for ASR
        if validated["audio"]:
            ctx["artifacts"]["primary_audio"] = validated["audio"][0]["processed_path"]
        elif validated["video"]:
            # Extract audio from video in next stage
            ctx["artifacts"]["primary_video"] = validated["video"][0]["processed_path"]
            ctx["artifacts"]["primary_audio"] = None  # Will be extracted by ASR stage
        
        self.logger.info(
            f"Ingestion complete: {sum(len(v) for v in validated.values())} files validated"
        )
        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_files(directory: Path, extensions: list[str]) -> list[Path]:
        """Recursively find files matching given extensions."""
        found = []
        for ext in extensions:
            found.extend(directory.rglob(f"*{ext}"))
        return sorted(set(found))

    @staticmethod
    def _get_media_duration(path: Path) -> float | None:
        """Get media file duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return float(info["format"]["duration"])
        except (KeyError, ValueError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
