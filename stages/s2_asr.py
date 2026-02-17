"""
Stage 2: Automatic Speech Recognition & Speaker Diarization
============================================================
- Uses trimmed audio window defined in inventory.json
- Runs Whisper (faster-whisper) for transcription
- Runs pyannote.audio for speaker diarization
- Merges ASR segments with speaker labels
- Outputs diarized transcript with original timestamps
"""

import json
import os
import subprocess
from pathlib import Path

from stages.base import BaseStage


class ASRStage(BaseStage):
    """Speech-to-text with speaker diarization."""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("asr")
        output_dir = Path(ctx["output_base"]) / "02_asr"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Read conversation window from inventory (from disk = source of truth)
        # ------------------------------------------------------------------
        inventory_path = Path(ctx["output_base"]) / "01_ingest" / "inventory.json"
        if inventory_path.exists():
            with open(inventory_path) as f:
                inventory = json.load(f)
            self.logger.info(f"[ASR] Loaded inventory from {inventory_path}")
        else:
            inventory = ctx["artifacts"].get("inventory", {})
            self.logger.warning("[ASR] inventory.json not found on disk, using ctx")
        audio_meta = inventory.get("audio", {})

        conversation_start = float(audio_meta.get("conversation_start_s", 0.0))
        conversation_end = audio_meta.get("conversation_end_s", None)

        # ------------------------------------------------------------------
        # Resolve audio source
        # ------------------------------------------------------------------
        audio_path = ctx["artifacts"].get("primary_audio")
        if audio_path is None:
            raise ValueError("No primary_audio found for ASR stage.")

        self.logger.info(f"Audio source: {audio_path}")

        # ------------------------------------------------------------------
        # Trim audio if needed
        # ------------------------------------------------------------------
        trimmed_audio = output_dir / "audio_for_asr.wav"

        if conversation_start > 0 or conversation_end is not None:
            self.logger.info(
                f"Trimming audio for ASR: start={conversation_start}s, "
                f"end={conversation_end}"
            )
            self._trim_audio(
                input_audio=audio_path,
                output_audio=str(trimmed_audio),
                start_s=conversation_start,
                end_s=conversation_end,
            )
            audio_for_asr = str(trimmed_audio)
        else:
            audio_for_asr = audio_path

        # ------------------------------------------------------------------
        # Whisper transcription
        # ------------------------------------------------------------------
        segments = self._transcribe(audio_for_asr, cfg)
        self.logger.info(f"Whisper produced {len(segments)} segments")

        # ------------------------------------------------------------------
        # Speaker diarization
        # ------------------------------------------------------------------
        diar_cfg = cfg.get("diarization", {})
        speaker_segments = None

        if diar_cfg.get("enabled", False):
            speaker_segments = self._diarize(audio_for_asr, diar_cfg)
            if speaker_segments:
                self.logger.info(
                    f"Diarization identified speakers in {len(speaker_segments)} segments"
                )

        # ------------------------------------------------------------------
        # Merge ASR + diarization
        # ------------------------------------------------------------------
        transcript = self._merge(segments, speaker_segments)

        # ------------------------------------------------------------------
        # Restore original timeline
        # ------------------------------------------------------------------
        if conversation_start > 0:
            for seg in transcript:
                seg["start"] += conversation_start
                seg["end"] += conversation_start

        # ------------------------------------------------------------------
        # Optional speaker post-processing
        # ------------------------------------------------------------------
        #if speaker_segments:
        #    transcript = self._postprocess_speakers(transcript)

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------
        transcript_path = output_dir / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        readable_path = output_dir / "transcript.txt"
        with open(readable_path, "w", encoding="utf-8") as f:
            for seg in transcript:
                start = self._fmt_time(seg["start"])
                end = self._fmt_time(seg["end"])
                speaker = seg.get("speaker", "?")
                f.write(f"[{start} → {end}] {speaker}: {seg['text']}\n")

        ctx["artifacts"]["transcript"] = transcript
        ctx["artifacts"]["transcript_path"] = str(transcript_path)
        ctx["artifacts"]["transcript_readable_path"] = str(readable_path)

        return ctx

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------
    def _transcribe(self, audio_path: str, cfg: dict) -> list[dict]:
        from faster_whisper import WhisperModel
        import torch

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Loading Whisper model: {cfg['model_name']} on {device}")

        model = WhisperModel(
            cfg["model_name"],
            device=device,
            compute_type=cfg.get("compute_type", "float16") if device == "cuda" else "int8",
        )

        segments_gen, info = model.transcribe(
            audio_path,
            language=cfg.get("language", "de"),
            beam_size=cfg.get("beam_size", 5),
            word_timestamps=True,
        )

        self.logger.info(
            f"Detected language: {info.language} "
            f"(prob={info.language_probability:.2f})"
        )

        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            })

        return segments

    # ------------------------------------------------------------------
    # Speaker diarization
    # ------------------------------------------------------------------
    def _diarize(self, audio_path: str, cfg: dict) -> list[dict] | None:
        from pyannote.audio import Pipeline
        import torch

        hf_token = os.environ.get(cfg.get("hf_token_env", "HF_TOKEN"))
        if not hf_token:
            self.logger.warning("HF_TOKEN not set; skipping diarization.")
            return None

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Loading diarization model: {cfg['model']} on {device}")

        pipeline = Pipeline.from_pretrained(
            cfg["model"],
            use_auth_token=hf_token,
        )

        if device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))

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
    # Speaker post-processing
    # ------------------------------------------------------------------
    def _postprocess_speakers(self, transcript: list[dict]) -> list[dict]:
        if not transcript:
            return transcript

        # Fill UNKNOWN
        for i, seg in enumerate(transcript):
            if seg["speaker"] == "UNKNOWN":
                prev_sp = transcript[i - 1]["speaker"] if i > 0 else None
                next_sp = transcript[i + 1]["speaker"] if i < len(transcript) - 1 else None
                if prev_sp and prev_sp != "UNKNOWN":
                    seg["speaker"] = prev_sp
                elif next_sp and next_sp != "UNKNOWN":
                    seg["speaker"] = next_sp

        # Fix A-B-A glitches
        for i in range(1, len(transcript) - 1):
            prev_sp = transcript[i - 1]["speaker"]
            curr_sp = transcript[i]["speaker"]
            next_sp = transcript[i + 1]["speaker"]
            duration = transcript[i]["end"] - transcript[i]["start"]

            if prev_sp == next_sp and curr_sp != prev_sp and duration < 2.0:
                transcript[i]["speaker"] = prev_sp

        return transcript

    # ------------------------------------------------------------------
    # Audio trimming
    # ------------------------------------------------------------------
    def _trim_audio(
        self,
        input_audio: str,
        output_audio: str,
        start_s: float,
        end_s: float | None,
    ):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", input_audio,
        ]

        if end_s is not None:
            duration = end_s - start_s
            cmd += ["-t", str(duration)]

        cmd += [
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    # ------------------------------------------------------------------
    # Merge ASR + diarization
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(asr_segments: list[dict], diar_segments: list[dict] | None) -> list[dict]:
        if not diar_segments:
            for seg in asr_segments:
                seg["speaker"] = "UNKNOWN"
            return asr_segments

        for seg in asr_segments:
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for dseg in diar_segments:
                overlap_start = max(seg["start"], dseg["start"])
                overlap_end = min(seg["end"], dseg["end"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg["speaker"]

            seg["speaker"] = best_speaker

        return asr_segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"