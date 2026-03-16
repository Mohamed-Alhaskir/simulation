"""
Stage 1: Data Ingestion & Preprocessing
=========================================
Handles the actual recording format:
  - ONE composite 4-quadrant video per session (2x2 grid)
  - Audio embedded in the video (room microphone)
  - No separate physio or eye-tracking files

This stage:
  1. Validates the video file
  2. Loads or auto-generates metadata.json
  3. Extracts the audio track (for ASR in Stage 2)
  4. Splits the composite video into individual quadrants (optional, for analysis)
  5. Produces an inventory of all processed artifacts
"""

import json
import shutil
import subprocess
from pathlib import Path

from stages.base import BaseStage
from utils.artifact_io import save_artifact
from utils.scenario_map import resolve_scenario_id


class DataIngestionStage(BaseStage):
    """Ingest composite simulation video and prepare for downstream stages."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("ingest")
        input_dir = Path(ctx["input_path"])
        output_dir = Path(ctx["output_base"]) / "01_ingest"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Scanning input directory: {input_dir}")

        # ---- Load or generate metadata ----
        metadata = self._load_metadata(input_dir, output_dir, ctx)

        # ---- Inject scenario_id from session map if not already present ----
        # This covers three cases:
        #   a) User-provided metadata.json already has scenario.id → no-op
        #   b) Auto-generated metadata found scenario from catalog → no-op
        #   c) Neither worked (new session, catalog missing) → map file fills the gap
        self._ensure_scenario_id(metadata, ctx)

        ctx["artifacts"]["metadata"] = metadata

        # ---- Find the composite video ----
        video_path = self._find_composite_video(input_dir, metadata, cfg)
        self.logger.info(f"Composite video: {video_path.name}")

        # ---- Validate video ----
        duration = self._get_media_duration(video_path)
        if duration is None:
            raise ValueError(f"Cannot read video duration: {video_path}")

        min_dur = cfg.get("min_duration_s", 30)
        max_dur = cfg.get("max_duration_s", 3600)
        if duration < min_dur or duration > max_dur:
            raise ValueError(
                f"Video duration {duration:.1f}s outside valid range "
                f"[{min_dur}, {max_dur}]s"
            )

        resolution = self._get_video_resolution(video_path)
        self.logger.info(
            f"Duration: {duration:.1f}s | Resolution: {resolution[0]}x{resolution[1]}"
        )

        # Copy video to processed directory
        processed_video = output_dir / video_path.name
        shutil.copy2(video_path, processed_video)

        # ---- Extract audio track ----
        audio_path = output_dir / "audio_extracted.wav"
        self._extract_audio(str(processed_video), str(audio_path))
        self.logger.info(f"Audio extracted: {audio_path.name}")

        # ---- Split quadrants (optional, for visual analysis) ----
        quadrants = {}
        composite_cfg = cfg.get("composite_video", {})
        if composite_cfg.get("enabled", False):
            quadrants = self._split_quadrants(
                str(processed_video), str(output_dir), resolution
            )
            self.logger.info(f"Split into {len(quadrants)} quadrants")

        # ---- Resolve which quadrant Stage 04 should analyse ----
        video_analysis_quadrant = (
            metadata.get("video_analysis_quadrant")
            or composite_cfg.get("video_analysis_quadrant", "bottom_left")
        )

        # ---- Build inventory ----
        inventory = {
            "composite_video": {
                "path": str(processed_video),
                "duration_s": duration,
                "resolution": resolution,
            },
            "audio": {
                "path": str(audio_path),
                "duration_s": duration,
                "conversation_start_s": 0.0,
                "conversation_end_s": duration,
            },
            "quadrants": quadrants,
            "video_analysis_quadrant": video_analysis_quadrant,
        }

        inventory_path = output_dir / "inventory.json"
        save_artifact(
            inventory, inventory_path,
            description="inventory", logger_instance=self.logger
        )

        # ---- Update context ----
        ctx["artifacts"]["inventory"]        = inventory
        ctx["artifacts"]["inventory_path"]   = str(inventory_path)
        ctx["artifacts"]["primary_audio"]    = str(audio_path)
        ctx["artifacts"]["composite_video"]  = str(processed_video)
        ctx["artifacts"]["video_duration_s"] = duration
        ctx["artifacts"]["video_resolution"] = resolution

        self.logger.info("Ingestion complete")
        return ctx

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(
        self, input_dir: Path, output_dir: Path, ctx: dict
    ) -> dict:
        """
        Load metadata from the first available source in priority order:

        1. output_dir/metadata.json  — previously processed run (idempotent re-runs)
        2. input_dir/metadata.json   — user-provided alongside the video file
        3. Auto-generate             — minimal stub from scenario catalog + video scan

        In all cases the loaded/generated dict is saved (or re-saved) to
        output_dir/metadata.json so downstream stages always find it there.
        """
        output_metadata_path = output_dir / "metadata.json"
        input_metadata_path  = input_dir  / "metadata.json"

        # ── Source 1: already processed ────────────────────────────────
        if output_metadata_path.exists():
            with open(output_metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            self.logger.info(
                f"Loaded existing metadata from output dir: "
                f"scenario={metadata.get('scenario', {}).get('id', 'N/A')}"
            )
            return metadata

        # ── Source 2: user-provided in input dir ───────────────────────
        if input_metadata_path.exists():
            with open(input_metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            self.logger.info(
                f"Loaded user-provided metadata from input dir: "
                f"scenario={metadata.get('scenario', {}).get('id', 'N/A')}"
            )
            # Save a copy to output dir so future re-runs hit Source 1
            save_artifact(
                metadata, output_metadata_path,
                description="metadata", logger_instance=self.logger
            )
            return metadata

        # ── Source 3: auto-generate ────────────────────────────────────
        self.logger.warning(
            "No metadata.json found in input or output dir — "
            "auto-generating from catalog..."
        )
        metadata = self._auto_generate_metadata(input_dir, ctx)
        save_artifact(
            metadata, output_metadata_path,
            description="metadata_generated", logger_instance=self.logger
        )
        self.logger.info(
            f"Auto-generated metadata saved to {output_metadata_path} "
            "— please review."
        )
        return metadata

    def _ensure_scenario_id(self, metadata: dict, ctx: dict) -> None:
        """
        Guarantee metadata["scenario"]["id"] is a non-empty canonical string.

        If scenario.id is already populated (from user-provided metadata or
        auto-generation via catalog), this is a no-op.

        If it is missing or empty, resolve it from the session_scenario_map
        and write it into the metadata dict in-place. The caller is responsible
        for saving the updated metadata to disk if persistence is needed.
        """
        existing = metadata.get("scenario", {}).get("id", "").strip()
        if existing:
            self.logger.debug(
                f"scenario_id already present: '{existing}' — skipping map lookup"
            )
            return

        session_id  = ctx.get("session_id", "")
        scenario_id = resolve_scenario_id(session_id)

        if scenario_id:
            metadata.setdefault("scenario", {})
            metadata["scenario"]["id"] = scenario_id
            self.logger.info(
                f"scenario_id injected from session map: "
                f"'{session_id}' → '{scenario_id}'"
            )
        else:
            self.logger.warning(
                f"Could not resolve scenario_id for session '{session_id}'. "
                f"metadata['scenario']['id'] remains empty. "
                f"Stage 5 will attempt a runtime fallback lookup."
            )

    def _auto_generate_metadata(self, input_dir: Path, ctx: dict) -> dict:
        """Generate minimal metadata when none is provided."""
        from datetime import date

        metadata = {
            "session_id": ctx["session_id"],
            "date": str(date.today()),
            "participants": [
                {"role": "learner",           "pseudonym": "PARTICIPANT_A"},
                {"role": "simulated_patient", "pseudonym": "SP_001"},
            ],
            "recordings": {
                "composite_video": None,
            },
            "duration_planned_min": 15,
            "site":  "Wuppertal",
            "notes": "Auto-generated metadata.",
            "scenario": {"id": ""},
        }

        # Look up scenario from catalog if available
        catalog_path = Path(self.config["paths"]["scenario_catalog"])
        self.logger.info(f"Looking for scenario catalog at {catalog_path}")
        if catalog_path.exists():
            with open(catalog_path, encoding="utf-8") as f:
                catalog = json.load(f)
            # Support both old flat structure and new nested structure
            sessions_map = catalog.get("sessions", catalog)
            scenario_id = sessions_map.get(ctx["session_id"], "")
            if scenario_id:
                from utils.scenario_map import _canonicalise
                metadata["scenario"]["id"] = _canonicalise(scenario_id)
                self.logger.info(
                    f"Scenario resolved from catalog: "
                    f"'{scenario_id}' → '{metadata['scenario']['id']}'"
                )
        else:
            self.logger.warning(
                f"Scenario catalog not found at {catalog_path}. "
                "scenario.id will be empty — update metadata.json manually."
            )

        # Auto-detect video
        videos = self._find_videos(input_dir)
        if videos:
            metadata["recordings"]["composite_video"] = videos[0].name

        return metadata

    # ------------------------------------------------------------------
    # Video discovery
    # ------------------------------------------------------------------

    def _find_composite_video(
        self, input_dir: Path, metadata: dict, cfg: dict
    ) -> Path:
        """Find the composite video file."""
        rec      = metadata.get("recordings", {})
        specified = rec.get("composite_video") or rec.get("primary_video")
        if specified:
            for candidate in [
                input_dir / specified,
                input_dir / "video" / specified,
            ]:
                if candidate.exists():
                    return candidate

        videos = self._find_videos(input_dir)
        if not videos:
            raise ValueError(
                f"No video files found in {input_dir}. "
                f"Accepted formats: {cfg.get('accepted_video_formats', [])}"
            )
        if len(videos) > 1:
            self.logger.warning(
                f"Multiple videos found, using first: {videos[0].name}"
            )
        return videos[0]

    def _find_videos(self, directory: Path) -> list[Path]:
        """Find all video files in directory."""
        cfg        = self._get_stage_config("ingest")
        extensions = cfg.get(
            "accepted_video_formats", [".mp4", ".avi", ".mkv", ".mov"]
        )
        found = []
        for ext in extensions:
            found.extend(directory.glob(f"*{ext}"))
            found.extend(directory.glob(f"video/*{ext}"))
        seen = {}
        for p in found:
            resolved = p.resolve()
            if resolved not in seen:
                seen[resolved] = p
        return sorted(seen.values())

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_audio(video_path: str, output_path: str):
        """
        Extract audio track with diarization-safe preprocessing:
        mono, 16 kHz, loudness normalisation (two-pass), high-pass filter.
        """
        # Pass 1: measure loudness statistics
        measure_result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-af", "highpass=f=70,loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json",
                "-f", "null", "-",
            ],
            capture_output=True, text=True,
        )
        measured = {}
        try:
            stderr     = measure_result.stderr
            json_start = stderr.rfind("{")
            json_end   = stderr.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                measured = json.loads(stderr[json_start:json_end])
        except (ValueError, KeyError):
            pass

        # Pass 2: apply normalisation using measured values
        if measured.get("input_i"):
            loudnorm_filter = (
                f"highpass=f=70,"
                f"loudnorm=I=-16:LRA=11:TP=-1.5"
                f":measured_I={measured['input_i']}"
                f":measured_LRA={measured['input_lra']}"
                f":measured_TP={measured['input_tp']}"
                f":measured_thresh={measured['input_thresh']}"
                f":offset={measured.get('target_offset', '0.0')}"
                f":linear=true"
            )
        else:
            loudnorm_filter = "highpass=f=70,loudnorm=I=-16:LRA=11:TP=-1.5"

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-af", loudnorm_filter,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path,
            ],
            check=True, capture_output=True,
        )

    # ------------------------------------------------------------------
    # Quadrant splitting
    # ------------------------------------------------------------------

    def _split_quadrants(
        self, video_path: str, output_dir: str, resolution: tuple[int, int]
    ) -> dict:
        """
        Split a 2x2 composite video into 4 individual quadrant videos.

          ┌────────────┬────────────┐
          │ top_left   │ top_right  │
          ├────────────┼────────────┤
          │ bottom_left│bottom_right│
          └────────────┴────────────┘
        """
        w, h    = resolution
        half_w  = w // 2
        half_h  = h // 2

        quadrant_crops = {
            "top_left":     f"crop={half_w}:{half_h}:0:0",
            "top_right":    f"crop={half_w}:{half_h}:{half_w}:0",
            "bottom_left":  f"crop={half_w}:{half_h}:0:{half_h}",
            "bottom_right": f"crop={half_w}:{half_h}:{half_w}:{half_h}",
        }

        quadrants = {}
        for name, crop_filter in quadrant_crops.items():
            out_path = f"{output_dir}/quadrant_{name}.mp4"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vf", crop_filter,
                        "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        out_path,
                    ],
                    check=True, capture_output=True,
                )
                quadrants[name] = out_path
            except subprocess.CalledProcessError as e:
                self.logger.warning(
                    f"Quadrant split failed for '{name}': "
                    f"{e.stderr.decode(errors='replace').strip()}"
                )
        return quadrants

    # ------------------------------------------------------------------
    # Media info helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_media_duration(path: Path) -> float | None:
        """Get media file duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json", "-show_format",
                    str(path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return float(info["format"]["duration"])
        except (KeyError, ValueError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _get_video_resolution(self, path: Path) -> tuple[int, int]:
        """Get video width and height using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json", "-show_streams",
                    "-select_streams", "v:0", str(path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                info   = json.loads(result.stdout)
                stream = info["streams"][0]
                return (int(stream["width"]), int(stream["height"]))
        except (KeyError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        self.logger.warning(
            f"ffprobe could not read resolution for {path}; "
            "falling back to 1920x1080."
        )
        return (1920, 1080)