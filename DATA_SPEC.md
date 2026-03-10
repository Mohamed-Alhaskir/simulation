# Data Specification

## Your Setup

Each session = **one composite video file** (2x2 grid, audio embedded):

```
┌────────────────────┬────────────────────┐
│  Overhead camera   │  Patient monitor   │
│  (top-left)        │  (top-right)       │
├────────────────────┼────────────────────┤
│  Side camera       │  Parent eye-track  │
│  (bottom-left)     │  (bottom-right)    │
└────────────────────┴────────────────────┘
```

No separate physio or eye-tracking data files. All data is captured in the
composite video. The top-right quadrant (patient monitor) can optionally be
processed via OCR to extract vitals.

## How to Set Up Your Data

```
data/raw/
├── session_001/
│   └── recording.mp4
├── session_002/
│   └── recording.mp4
├── session_003/
│   └── recording.mp4
└── session_004/
    └── recording.mp4
```

One folder per session, one video file inside. Any `.mp4`/`.avi`/`.mkv`/`.mov`
filename works.

Optionally place a `metadata.json` in the session folder to specify the scenario
and participants. If absent, the pipeline auto-generates one from the scenario
catalog.

## Running

```bash
# All sessions
python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch

# Single session
python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/

# Force re-run (ignore stage checkpoints)
python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/ --force
```

## What the Pipeline Produces

```
data/reports/session_001/
├── 01_ingest/
│   ├── recording.mp4                  # Copy of original
│   ├── audio_extracted.wav            # 16kHz mono, loudness-normalised (for Whisper)
│   ├── quadrant_top_left.mp4          # Overhead camera
│   ├── quadrant_top_right.mp4         # Patient monitor
│   ├── quadrant_bottom_left.mp4       # Side camera
│   ├── quadrant_bottom_right.mp4      # Parent eye-tracking view
│   ├── metadata.json                  # Session metadata (loaded or auto-generated)
│   └── inventory.json                 # Paths + durations for all produced artifacts
│
├── 02_asr/
│   ├── audio_for_asr.wav              # Trimmed audio sent to Whisper (if windowed)
│   ├── transcript.json                # Diarized transcript with timestamps
│   └── transcript.txt                 # Human-readable transcript
│
├── 03_features/
│   └── features.json                  # Turn-taking, pauses, interruptions,
│                                      # speaker ratios, response latencies,
│                                      # conversation phase segmentation
│
├── 04_video_analysis/
│   ├── video_features.json            # MediaPipe NVB metrics: eye contact,
│   │                                  # posture, facial expressions, gestures
│   │                                  # (person-relative, distribution-based)
│   └── annotated_video.mp4            # Optional: frame-annotated video
│
├── 05_analysis/
│   ├── assembled_context.json         # Reproducibility ground truth —
│   │                                  # exact input fed to both LLM passes
│   ├── spikes_prompt.txt              # Pass 1: SPIKES annotation prompt
│   ├── spikes_raw_output.txt          # Pass 1: raw LLM output
│   ├── spikes_annotation.json         # Pass 1: parsed SPIKES step annotation
│   ├── lucas_prompt.txt               # Pass 2: LUCAS scoring prompt
│   ├── lucas_raw_output.txt           # Pass 2: raw LLM output
│   ├── lucas_analysis.json            # Pass 2: LUCAS items A-J with scores,
│   │                                  # justifications, and evidence (max 18)
│   ├── clinical_content_prompt.txt    # Pass 3: clinical content check prompt
│   ├── clinical_content_raw_output.txt
│   └── clinical_content.json          # Pass 3: clinical accuracy validation
│
├── 07_report/
│   ├── REPORT_session_001.json        # Final structured report
│   ├── REPORT_session_001.html        # Printable / browser version
│   └── REPORT_session_001.pdf         # PDF (requires weasyprint)
│
└── pipeline_meta.json                 # Run timestamps, stage durations,
                                       # freeze manifest hash
```

## Stage Checkpoints

Each stage writes a `.stage_checkpoint.json` file inside its output directory
once it completes successfully. On re-run, completed stages are skipped. Use
`--force` to clear all checkpoints and re-run from scratch.
