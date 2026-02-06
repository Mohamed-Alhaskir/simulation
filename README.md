# Paediatric Simulation AI Feedback Pipeline

Frozen, version-locked pipeline for generating standardized feedback reports from audiovisual recordings of paediatric simulation scenarios.

> **Study context**: This pipeline is part of a prospective evaluation at University Witten/Herdecke (Ethics ref: S-44/2025). It augments — rather than replaces — instructor-led debriefings by producing structured, reproducible feedback reports.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FREEZE MANIFEST                          │
│  (commit hash, model versions, prompts, seeds, timestamp)    │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  01 INGEST   │────▶│   02 ASR     │────▶│  03 FEATURES     │
│  Video/Audio │     │  Whisper +   │     │  Turn-taking,    │
│  Physio/Eye  │     │  Diarization │     │  Physio, Gaze    │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                   │
                                                   ▼
                     ┌──────────────┐     ┌──────────────────┐
                     │  05 REPORT   │◀────│  04 LLM ANALYSIS │
                     │  JSON / HTML │     │  Local Llama     │
                     │  (Blinded)   │     │  3 Domains       │
                     └──────────────┘     └──────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

**Whisper** (downloaded automatically by faster-whisper on first run)

**LLM** — download a GGUF model:
```bash
mkdir -p models/
# Example: Llama 3.1 8B Instruct (quantized)
# Download from https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
wget -O models/llama-3.1-8b-instruct.gguf <DOWNLOAD_URL>
```

**Pyannote** — requires a HuggingFace token:
```bash
export HF_TOKEN=hf_your_token_here
# Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
```

### 3. Prepare session data

```
data/raw/session_001/
├── recording.mp4          # Simulation video
├── physio.csv             # Optional: HR, EDA data
└── eyetracking.csv        # Optional: gaze data
```

### 4. Run the pipeline

```bash
# Single session
python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/

# Batch mode (all sessions)
python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch

# Print freeze manifest
python pipeline.py --freeze-manifest
```

## Output Structure

```
data/reports/session_001/
├── 01_ingested/          # Validated input files + inventory
├── 02_asr/               # Transcript (JSON + readable TXT)
├── 03_features/          # Extracted features + LLM profile
├── 04_analysis/          # LLM prompt, raw output, parsed analysis
├── 05_report/            # Final report (JSON, HTML, PDF)
└── pipeline_meta.json    # Run metadata + timing
```

## Freeze Protocol

Before any confirmatory data analysis:

1. Run `python pipeline.py --freeze-manifest > freeze_manifest.json`
2. Commit all code and archive the manifest
3. Do NOT modify code, prompts, model weights, or config after freeze
4. The pipeline verifies manifest integrity on each run

## Configuration

All parameters are in `config/pipeline_config.yaml`:

| Section | Key settings |
|---------|-------------|
| `asr` | Whisper model, beam size, diarization on/off |
| `llm` | Model path, temperature (0.0), seed, context length |
| `features` | Pause threshold, HRV window, fixation threshold |
| `report` | Output formats, blinding labels |
| `evaluation` | Phase 1/2 thresholds (for reference, not used in generation) |

## Assessment Domains

| Domain | Description | Scale |
|--------|-------------|-------|
| Global Communication | Empathy, clarity, rapport | 1–5 |
| Conversation Structuring | Opening, agenda, transitions, closing | 1–5 |
| Clinical Content | Medical accuracy, history, reasoning | 1–5 |

## Project Structure

```
paed-sim-pipeline/
├── pipeline.py                 # Main orchestrator
├── config/
│   └── pipeline_config.yaml    # All configuration
├── stages/
│   ├── base.py                 # Abstract base stage
│   ├── s1_ingest.py            # Data ingestion & validation
│   ├── s2_asr.py               # Whisper + diarization
│   ├── s3_features.py          # Feature extraction
│   ├── s4_analysis.py          # LLM analysis
│   └── s5_report.py            # Report generation
├── utils/
│   ├── freeze.py               # Freeze manifest
│   └── logging_setup.py        # Logging config
├── templates/
│   └── analysis_prompt.j2      # LLM prompt template (German)
├── requirements.txt
└── README.md
```
