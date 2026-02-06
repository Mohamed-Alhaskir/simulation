"""
Stage 2: Automatic Speech Recognition & Speaker Diarization
============================================================
- Extracts audio from video if needed (ffmpeg)
- Runs Whisper (faster-whisper) for German transcription
- Runs pyannote speaker diarization
- Merges ASR segments with speaker labels
- Outputs a diarized transcript with timestamps
"""

import json
import os
import subprocess
from pathlib import Path

from stages.base import BaseStage


class ASRStage(BaseStage):
    """Speech-to-text with speaker diarization."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("asr")
        output_dir = Path(ctx["output_base"]) / "02_asr"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve audio source
        audio_path = ctx["artifacts"].get("primary_audio")
        if audio_path is None:
            video_path = ctx["artifacts"].get("primary_video")
            if video_path is None:
                raise ValueError("No audio or video source available for ASR.")
            audio_path = str(output_dir / "extracted_audio.wav")
            self._extract_audio(video_path, audio_path)
            ctx["artifacts"]["primary_audio"] = audio_path

        self.logger.info(f"Audio source: {audio_path}")

        # ---- Whisper transcription ----
        segments = self._transcribe(audio_path, cfg)
        self.logger.info(f"Whisper produced {len(segments)} segments")

        # ---- Speaker diarization ----
        diar_cfg = cfg.get("diarization", {})
        speaker_segments = None
        if diar_cfg.get("enabled", False):
            speaker_segments = self._diarize(audio_path, diar_cfg)
            self.logger.info(
                f"Diarization identified speakers in {len(speaker_segments)} segments"
            )

        # ---- Merge transcription + diarization ----
        transcript = self._merge(segments, speaker_segments)

        # ---- Save outputs ----
        transcript_path = output_dir / "transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        # Also save a human-readable version
        readable_path = output_dir / "transcript.txt"
        with open(readable_path, "w") as f:
            for seg in transcript:
                speaker = seg.get("speaker", "?")
                start = self._fmt_time(seg["start"])
                end = self._fmt_time(seg["end"])
                f.write(f"[{start} → {end}] {speaker}: {seg['text']}\n")

        ctx["artifacts"]["transcript"] = transcript
        ctx["artifacts"]["transcript_path"] = str(transcript_path)
        ctx["artifacts"]["transcript_readable_path"] = str(readable_path)

        return ctx

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------
    def _transcribe(self, audio_path: str, cfg: dict) -> list[dict]:
        """
        Transcribe audio using faster-whisper.

        Returns list of dicts with keys: start, end, text.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.logger.error(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
            raise

        device = cfg.get("device", "auto")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.logger.info(
            f"Loading Whisper model: {cfg['model_name']} on {device}"
        )
        model = WhisperModel(
            cfg["model_name"],
            device=device,
            compute_type=cfg.get("compute_type", "float16") if device != "cpu" else "int8",
        )

        segments_gen, info = model.transcribe(
            audio_path,
            language=cfg.get("language", "de"),
            beam_size=cfg.get("beam_size", 5),
            word_timestamps=True,
        )

        self.logger.info(
            f"Detected language: {info.language} (prob={info.language_probability:.2f})"
        )

        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (seg.words or [])
                ],
            })

        return segments

    # ------------------------------------------------------------------
    # Speaker diarization
    # ------------------------------------------------------------------
    def _diarize(self, audio_path: str, cfg: dict) -> list[dict]:
        """
        Run speaker diarization using pyannote.audio.

        Returns list of dicts with keys: start, end, speaker.
        """
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            self.logger.warning(
                "pyannote.audio not installed. Skipping diarization. "
                "Install with: pip install pyannote.audio"
            )
            return None

        hf_token = os.environ.get(cfg.get("hf_token_env", "HF_TOKEN"))
        if not hf_token:
            self.logger.warning(
                "HuggingFace token not found. Diarization requires access to "
                "pyannote models. Set HF_TOKEN environment variable."
            )
            return None

        self.logger.info(f"Loading diarization model: {cfg['model']}")
        pipeline = PyannotePipeline.from_pretrained(
            cfg["model"],
            use_auth_token=hf_token,
        )

        diarization = pipeline(
            audio_path,
            min_speakers=cfg.get("min_speakers", 2),
            max_speakers=cfg.get("max_speakers", 5),
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        return segments

    # ------------------------------------------------------------------
    # Merge transcription + diarization
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(
        asr_segments: list[dict],
        diar_segments: list[dict] | None,
    ) -> list[dict]:
        """
        Assign speaker labels to ASR segments based on temporal overlap
        with diarization output.
        """
        if not diar_segments:
            # No diarization — return transcript without speaker labels
            for seg in asr_segments:
                seg["speaker"] = "UNKNOWN"
            return asr_segments

        for seg in asr_segments:
            mid = (seg["start"] + seg["end"]) / 2
            best_speaker = "UNKNOWN"
            best_overlap = 0

            for dseg in diar_segments:
                # Calculate overlap
                overlap_start = max(seg["start"], dseg["start"])
                overlap_end = min(seg["end"], dseg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg["speaker"]

            seg["speaker"] = best_speaker

        return asr_segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_audio(video_path: str, output_path: str):
        """Extract audio track from video using ffmpeg."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                  # No video
                "-acodec", "pcm_s16le", # WAV format
                "-ar", "16000",         # 16kHz (Whisper optimal)
                "-ac", "1",             # Mono
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
