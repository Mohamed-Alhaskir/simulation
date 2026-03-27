"""
Stage 2: Automatic Speech Recognition & Speaker Diarization
============================================================
- Uses trimmed audio window defined in inventory.json
- Delegates to whisper-diarization (MahmoudAshraf97) for transcription + diarization
  which uses: Demucs source separation → Whisper + CTC alignment →
              NeMo TitaNet speaker embeddings → word-level speaker assignment
- Parses whisper-diarization output back into pipeline transcript format
- Outputs diarized transcript with original timestamps

Config:
  asr:
    model_name: large-v3
    language: de
    batch_size: 8
    device: cuda
    suppress_numerals: True
    diarization:
      enabled: true
      repo_path: /path/to/whisper-diarization   # clone of MahmoudAshraf97/whisper-diarization
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from stages.base import BaseStage


class ASRStage(BaseStage):
    """Speech-to-text with speaker diarization via whisper-diarization."""

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
        # Transcription + diarization
        # ------------------------------------------------------------------
        diar_cfg = cfg.get("diarization", {})

        if diar_cfg.get("enabled", False):
            self.logger.info("Running whisper-diarization pipeline...")
            transcript = self._run_whisper_diarization(audio_for_asr, cfg, diar_cfg, output_dir)
            self.logger.info(f"Got {len(transcript)} segments from whisper-diarization")
        else:
            self.logger.info("Diarization disabled — transcribing without speaker labels...")
            transcript = self._transcribe_only(audio_for_asr, cfg)
            for seg in transcript:
                seg["speaker"] = "UNKNOWN"

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
    # Run whisper-diarization as subprocess + parse output
    # ------------------------------------------------------------------
    def _run_whisper_diarization(
        self,
        audio_path: str,
        cfg: dict,
        diar_cfg: dict,
        output_dir: Path,
    ) -> list[dict]:
        import torch

        repo_path = diar_cfg.get("repo_path")
        if not repo_path or not Path(repo_path).exists():
            raise ValueError(
                f"whisper-diarization repo not found at '{repo_path}'. "
                "Clone https://github.com/MahmoudAshraf97/whisper-diarization "
                "and set diarization.repo_path in config."
            )
        device = cfg.get("device", "cuda")

        # whisper-diarization writes output next to the input file
        # so we work in a temp dir to keep things clean
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy audio into temp dir
            tmp_audio = Path(tmpdir) / "input.wav"
            subprocess.run(
                ["cp", audio_path, str(tmp_audio)],
                check=True
            )

            cmd = [
                "conda", "run",
                "-n", "whisper-diarization",
                "python", "diarize.py",
                "-a", str(tmp_audio),
                "--whisper-model", cfg.get("model_name", "large-v3"),
                "--device", device,
                "--language", cfg.get("language", "de"),
                "--batch-size", str(diar_cfg.get("batch_size", 16)),
                "--beam-size", str(cfg.get("beam_size", 7)),
                "--temperature", str(diar_cfg.get("temperature", 0)),
            ]

            if cfg.get("suppress_numerals", True):
                cmd.append("--suppress_numerals")

            if diar_cfg.get("no_stem", True):
                cmd.append("--no-stem")

            self.logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=False,   # show output in logs
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"whisper-diarization failed with return code {result.returncode}"
                )

            # whisper-diarization writes a .txt file next to the audio
            # Format: "Speaker X: text\n" with optional timestamps
            txt_output = Path(tmpdir) / "input.txt"
            srt_output = Path(tmpdir) / "input.srt"

            if srt_output.exists():
                self.logger.info("Parsing SRT output (real timestamps + speakers)...")
                transcript = self._parse_srt(str(srt_output))
            elif txt_output.exists():
                self.logger.info("Parsing TXT output (no timestamps, fallback)...")
                transcript = self._parse_txt(str(txt_output))
            else:
                # Search for any output file
                outputs = list(Path(tmpdir).glob("input*"))
                self.logger.warning(f"Expected output not found. Files: {outputs}")
                raise RuntimeError(
                    "whisper-diarization did not produce expected output file. "
                    f"Files found: {outputs}"
                )

            # Copy outputs to output_dir for reference
            for f in Path(tmpdir).glob("input*"):
                dest = output_dir / f"whisper_diarization_{f.suffix.lstrip('.')}"
                subprocess.run(["cp", str(f), str(dest)])

        return transcript

    # ------------------------------------------------------------------
    # Parse SRT output from whisper-diarization
    # SRT format:
    #   1
    #   00:00:01,000 --> 00:00:03,500
    #   Speaker 0: Hello, how are you?
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_srt(srt_path: str) -> list[dict]:
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        # Split into blocks
        blocks = re.split(r"\n\n+", content.strip())
        segments = []

        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue

            # Line 0: sequence number (skip)
            # Line 1: timestamps
            # Line 2+: "Speaker X: text"

            # Parse timestamps
            ts_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                lines[1]
            )
            if not ts_match:
                continue

            g = ts_match.groups()
            start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
            end   = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000

            # Parse speaker + text from remaining lines
            text_lines = lines[2:]
            full_text = " ".join(text_lines).strip()

            speaker_match = re.match(r"^(Speaker\s+\w+)\s*:\s*(.+)$", full_text, re.DOTALL)
            if speaker_match:
                raw_speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
                # Normalise: "Speaker 0" → "SPEAKER_00", "Speaker 1" → "SPEAKER_01"
                num_match = re.search(r"(\d+)", raw_speaker)
                if num_match:
                    n = int(num_match.group(1))
                    speaker = f"SPEAKER_{n:02d}"
                else:
                    speaker = raw_speaker.upper().replace(" ", "_")
            else:
                text = full_text
                speaker = "UNKNOWN"

            if not text:
                continue

            segments.append({
                "start":   start,
                "end":     end,
                "text":    text,
                "speaker": speaker,
                "words":   [],  # whisper-diarization txt/srt doesn't export word timestamps
            })

        return segments

    # ------------------------------------------------------------------
    # Parse plain TXT output from whisper-diarization
    # Format: "Speaker 0: text\n Speaker 1: text\n" (no timestamps)
    # Falls back to approximate timestamps from segment index
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_txt(txt_path: str) -> list[dict]:
        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()

        segments = []
        t = 0.0  # approximate running time — no timestamps in txt output

        for line in lines:
            line = line.strip()
            if not line:
                continue

            speaker_match = re.match(r"^(Speaker\s+\w+)\s*:\s*(.+)$", line)
            if speaker_match:
                raw_speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
                num_match = re.search(r"(\d+)", raw_speaker)
                if num_match:
                    n = int(num_match.group(1))
                    speaker = f"SPEAKER_{n:02d}"
                else:
                    speaker = raw_speaker.upper().replace(" ", "_")
            else:
                text = line
                speaker = "UNKNOWN"

            if not text:
                continue

            # Estimate duration from word count (~2.5 words/sec)
            word_count = len(text.split())
            duration = max(1.0, word_count / 2.5)

            segments.append({
                "start":   round(t, 3),
                "end":     round(t + duration, 3),
                "text":    text,
                "speaker": speaker,
                "words":   [],
            })
            t += duration

        return segments

    # ------------------------------------------------------------------
    # Fallback: Whisper-only transcription (no diarization)
    # ------------------------------------------------------------------
    def _transcribe_only(self, audio_path: str, cfg: dict) -> list[dict]:
        from faster_whisper import WhisperModel
        import torch

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = cfg.get("compute_type", "float16") if device == "cuda" else "int8"

        if not hasattr(self, "_whisper_model") or self._whisper_model is None:
            self.logger.info(f"Loading Whisper model: {cfg['model_name']} on {device}")
            try:
                self._whisper_model = WhisperModel(
                    cfg["model_name"], device=device, compute_type=compute_type
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    self.logger.warning("CUDA OOM — retrying on CPU.")
                    self._whisper_model = WhisperModel(
                        cfg["model_name"], device="cpu", compute_type="int8"
                    )
                else:
                    raise

        segments_gen, info = self._whisper_model.transcribe(
            audio_path,
            language=cfg.get("language", "de"),
            beam_size=cfg.get("beam_size", 7),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )

        self.logger.info(
            f"Detected language: {info.language} (prob={info.language_probability:.2f})"
        )

        segments = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue
            segments.append({
                "start": seg.start,
                "end":   seg.end,
                "text":  text,
                "words": [
                    {
                        "word":        w.word,
                        "start":       w.start,
                        "end":         w.end,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            })

        return segments

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
        cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-i", input_audio]
        if end_s is not None:
            cmd += ["-t", str(end_s - start_s)]
        cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio]
        subprocess.run(cmd, check=True, capture_output=True)

    def cleanup(self) -> None:
        if hasattr(self, "_whisper_model"):
            del self._whisper_model
            self._whisper_model = None
        import gc; gc.collect()

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"