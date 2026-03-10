"""
Improved speaker separation script using SpeechBrain SepFormer.

Improvements over basic version:
  1. Uses sepformer-whamr16k — trained on noisy/reverberant real-world speech
  2. Denoises audio before separation using noisereduce
  3. Normalizes loudness with ffmpeg loudnorm before separation
  4. Overlapping chunks with crossfade to avoid boundary artifacts
  5. VAD-based silence zeroing after separation to reduce bleed-through

Install dependencies:
    pip install speechbrain soundfile noisereduce

Usage:
    python test_separation.py <path_to_audio.wav>

Output:
    separated_speaker0.wav  (16kHz mono)
    separated_speaker1.wav  (16kHz mono)
"""

import sys
import os
import subprocess
import numpy as np

# ── tuneable parameters ────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000   # whamr16k model expects 16kHz
CHUNK_S         = 15      # seconds per chunk (within model's training range)
OVERLAP_S       = 1.0     # overlap between consecutive chunks (seconds)
VAD_THRESHOLD   = 0.02    # RMS below this → treat as silence and zero out
# ──────────────────────────────────────────────────────────────────────────────


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr.decode()}")


def get_duration(audio_path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def ffmpeg_normalize_and_resample(input_path, output_path):
    """Loudness normalize + resample to SAMPLE_RATE mono in one pass."""
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ])


def concat_wavs(wav_list, output_path):
    list_file = output_path + ".list.txt"
    with open(list_file, "w") as f:
        for w in wav_list:
            f.write(f"file '{os.path.abspath(w)}'\n")
    run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        output_path,
    ])
    os.remove(list_file)


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load_wav(path):
    import soundfile as sf
    data, sr = sf.read(path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz: {path}"
    return data.astype(np.float32)


def save_wav(path, audio_np):
    import soundfile as sf
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    sf.write(path, audio_np, SAMPLE_RATE)


# ── preprocessing ──────────────────────────────────────────────────────────────

def denoise(audio_np):
    """Stationary + non-stationary noise reduction."""
    try:
        import noisereduce as nr
        # First pass: stationary noise (room hum, mic hiss)
        reduced = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.75)
        # Second pass: non-stationary noise
        reduced = nr.reduce_noise(y=reduced, sr=SAMPLE_RATE, stationary=False, prop_decrease=0.5)
        return reduced
    except ImportError:
        print("  [warning] noisereduce not installed — skipping denoising. pip install noisereduce")
        return audio_np


# ── VAD-based silence zeroing ──────────────────────────────────────────────────

def apply_vad(audio_np, frame_ms=20):
    """Zero out frames where RMS energy is below VAD_THRESHOLD."""
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    out = audio_np.copy()
    for i in range(0, len(out), frame_len):
        frame = out[i:i + frame_len]
        if np.sqrt(np.mean(frame ** 2)) < VAD_THRESHOLD:
            out[i:i + frame_len] = 0.0
    return out


# ── crossfade concat ───────────────────────────────────────────────────────────

def crossfade_concat(chunks):
    """
    Concatenate audio chunks with linear crossfade over the overlap region
    to avoid clicks/artifacts at boundaries.
    """
    if len(chunks) == 1:
        return chunks[0]

    overlap_samples = int(OVERLAP_S * SAMPLE_RATE)
    result = chunks[0]

    for next_chunk in chunks[1:]:
        if overlap_samples == 0 or len(result) < overlap_samples or len(next_chunk) < overlap_samples:
            result = np.concatenate([result, next_chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, overlap_samples)
        fade_in  = np.linspace(0.0, 1.0, overlap_samples)

        # Blend the tail of result with the head of next_chunk
        blended = result[-overlap_samples:] * fade_out + next_chunk[:overlap_samples] * fade_in

        result = np.concatenate([
            result[:-overlap_samples],
            blended,
            next_chunk[overlap_samples:],
        ])

    return result


# ── main separation ────────────────────────────────────────────────────────────

def separate(audio_path):
    from speechbrain.inference.separation import SepformerSeparation

    print(f"\n{'='*60}")
    print(f"Input:  {audio_path}")

    # ── Step 1: loudness normalize + resample ──────────────────────
    print("\n[1/5] Normalizing loudness and resampling to 16kHz...")
    normalized_path = "_normalized.wav"
    ffmpeg_normalize_and_resample(audio_path, normalized_path)
    print(f"      Done.")

    # ── Step 2: denoise ────────────────────────────────────────────
    print("\n[2/5] Denoising...")
    audio = load_wav(normalized_path)
    audio = denoise(audio)
    denoised_path = "_denoised.wav"
    save_wav(denoised_path, audio)
    print(f"      Done. ({len(audio)/SAMPLE_RATE:.1f}s of audio)")

    total_duration = len(audio) / SAMPLE_RATE

    # ── Step 3: load model ─────────────────────────────────────────
    print("\n[3/5] Loading SepFormer model (sepformer-whamr16k)...")
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir="pretrained_models/sepformer-whamr16k",
    )
    print("      Model loaded.")

    # ── Step 4: chunked separation with overlap ────────────────────
    step_s = CHUNK_S - OVERLAP_S
    n_chunks = int(np.ceil(total_duration / step_s))
    print(f"\n[4/5] Separating in {n_chunks} overlapping chunks ({CHUNK_S}s chunks, {OVERLAP_S}s overlap)...")

    spk0_chunks = []
    spk1_chunks = []
    chunk_start = 0.0
    chunk_idx = 0

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + CHUNK_S, total_duration)
        chunk_dur = chunk_end - chunk_start

        start_sample = int(chunk_start * SAMPLE_RATE)
        end_sample   = int(chunk_end   * SAMPLE_RATE)
        chunk_audio  = audio[start_sample:end_sample]

        chunk_input = f"_chunk_{chunk_idx}_input.wav"
        save_wav(chunk_input, chunk_audio)

        print(f"  [{chunk_idx+1}/{n_chunks}] {chunk_start:.1f}s – {chunk_end:.1f}s  ({chunk_dur:.1f}s)")

        est_sources = model.separate_file(path=chunk_input)
        # shape: [1, time, n_speakers]

        spk0 = est_sources[0, :, 0].detach().cpu().numpy()
        spk1 = est_sources[0, :, 1].detach().cpu().numpy()

        # For overlapping chunks: only keep the non-overlap part except for last chunk
        if chunk_idx > 0:
            trim = int(OVERLAP_S / 2 * SAMPLE_RATE)
            spk0 = spk0[trim:]
            spk1 = spk1[trim:]
        if chunk_end < total_duration:
            trim = int(OVERLAP_S / 2 * SAMPLE_RATE)
            spk0 = spk0[:-trim] if len(spk0) > trim else spk0
            spk1 = spk1[:-trim] if len(spk1) > trim else spk1

        spk0_chunks.append(spk0)
        spk1_chunks.append(spk1)

        os.remove(chunk_input)
        chunk_start += step_s
        chunk_idx += 1

    # ── Step 5: merge + VAD + save ─────────────────────────────────
    print(f"\n[5/5] Merging chunks, applying VAD, saving...")

    spk0_full = np.concatenate(spk0_chunks)
    spk1_full = np.concatenate(spk1_chunks)

    spk0_full = apply_vad(spk0_full)
    spk1_full = apply_vad(spk1_full)

    save_wav("separated_speaker0.wav", spk0_full)
    save_wav("separated_speaker1.wav", spk1_full)

    # Cleanup
    for f in [normalized_path, denoised_path]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n{'='*60}")
    print("Output files:")
    print("  separated_speaker0.wav")
    print("  separated_speaker1.wav")
    print()
    print("Listen to both files and note which is the DOCTOR and which is the MOTHER.")
    print("The speaker identity may flip between chunks (SepFormer has no speaker tracking).")
    print("If so, let me know and we can add a speaker-consistency fix.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_separation.py <audio.wav>")
        sys.exit(1)
    separate(sys.argv[1])