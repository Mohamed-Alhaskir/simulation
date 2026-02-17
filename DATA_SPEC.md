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

That's it. One folder per session, one video file inside. Any `.mp4`/`.avi`/`.mkv`/`.mov` filename works.

## Running

```bash
# All 4 sessions
python pipeline.py --input data/raw/ --batch

# Single session
python pipeline.py --input data/raw/session_001/
```

## What the Pipeline Produces

```
data/reports/session_001/
├── 01_ingested/
│   ├── recording.mp4              # Copy of original
│   ├── audio_extracted.wav        # 16kHz mono (for Whisper)
│   ├── quadrant_top_left.mp4      # Overhead camera
│   ├── quadrant_top_right.mp4     # Patient monitor
│   ├── quadrant_bottom_left.mp4   # Side camera
│   ├── quadrant_bottom_right.mp4  # Parent eye-tracking view
│   ├── metadata.json  
│   └── inventory.json
├── 02_asr/
│   ├── transcript.json            # Diarized transcript
│   ├── audio_for asr.wav            # Diarized transcript
│   └── transcript.txt             # Human-readable
├── 03_video_analysis/
│   ├── features.json              # Turn-taking, pauses, interruptions
│   └── llm_profile.json           # Assembled input for LLM
├── 04_analysis/
│   ├── assembled_prompt.txt       # Exact prompt sent to LLM
│   ├── llm_raw_output.txt         # Raw model response
│   └── analysis.json              # Parsed ratings + feedback
├── 05_report/
│   ├── REPORT_session_001.json    # Final report
│   └── REPORT_session_001.html    # Printable version
└── pipeline_meta.json             # Timing + freeze manifest hash
```
