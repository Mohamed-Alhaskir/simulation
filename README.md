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
│  Extraction  │     │  Diarization │     │  Phase segments  │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  07 REPORT   │◀────│  05 ANALYSIS │◀────│ 04 VIDEO ANALYSIS│
│  JSON / HTML │     │  LUCAS +     │     │  MediaPipe NVB   │
│  PDF         │     │  SPIKES LLM  │     │  Face/Pose/Hands │
└──────────────┘     └──────────────┘     └──────────────────┘
```

Stage 06 (translation) exists but is disabled by default.

## Quick Start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate paed-sim-pipeline
```

### 2. Download models

**Whisper** (downloaded automatically by faster-whisper on first run)

**LLM** — download a GGUF model (requires `huggingface-hub`):
```bash
bash download_models.sh
# Recommended: Qwen2.5-32B-Instruct Q8_0 (~34 GB VRAM)
# Alternative: Qwen2.5-72B-Instruct Q4_K_M (~42 GB VRAM)
```

**Pyannote** — requires a HuggingFace token:
```bash
export HF_TOKEN=hf_your_token_here
# Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
```

### 3. Prepare session data

```
data/raw/session_001/
└── recording.mp4          # Composite 4-quadrant video with embedded audio
```

No separate physio or eye-tracking files — all data is captured in the composite video.

### 4. Configure model path

Edit `config/pipeline_config.yaml`:
```yaml
llm:
  model_path: "models/qwen2.5-32b-instruct-q8_0.gguf"
```

### 5. Run the pipeline

```bash
# Single session
python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/

# Batch mode (all sessions)
python pipeline.py --config config/pipeline_config.yaml --input data/raw/ --batch

# Force re-run (ignore checkpoints)
python pipeline.py --config config/pipeline_config.yaml --input data/raw/session_001/ --force

# Print freeze manifest
python pipeline.py --config config/pipeline_config.yaml --freeze-manifest
```

## Output Structure

```
data/reports/session_001/
├── 01_ingest/           # Validated inputs, extracted audio, quadrant clips
├── 02_asr/              # Transcript (JSON + readable TXT)
├── 03_features/         # Verbal interaction features + phase segmentation
├── 04_video_analysis/   # MediaPipe non-verbal behaviour features
├── 05_analysis/         # SPIKES + LUCAS LLM outputs, assembled context
├── 07_report/           # Final report (JSON, HTML, PDF)
└── pipeline_meta.json   # Run metadata + timing
```

See [DATA_SPEC.md](DATA_SPEC.md) for the full file listing within each directory.

## Freeze Protocol

Before any confirmatory data analysis:

1. Run `python pipeline.py --config config/pipeline_config.yaml --freeze-manifest > freeze_manifest.json`
2. Commit all code and archive the manifest
3. Do NOT modify code, prompts, model weights, or config after freeze
4. The pipeline verifies manifest integrity on each run

## Configuration

All parameters are in `config/pipeline_config.yaml`:

| Section | Key settings |
|---------|-------------|
| `asr` | Whisper model, device, beam size, diarization on/off |
| `llm` | Backend (`llama_cpp`/`vllm`), model path, temperature (0.0), seed, context length |
| `features` | Pause threshold, monitor OCR on/off |
| `video_analysis` | Enabled, sample FPS |
| `report` | Output formats (json/html/pdf), blinding labels |
| `evaluation` | Phase 1/2 thresholds (reference only, not used in generation) |

## Assessment Framework

### LUCAS (University of Liverpool Communication Assessment Scale)

10 items scored A–J, maximum total 18 points:

| Item | Description | Max |
|------|-------------|-----|
| A | Greeting and introduction | 1 |
| B | Identity check | 1 |
| C | Audibility and clarity of speech | 2 |
| D | Non-verbal behaviour | 2 |
| E | Questions, prompts, and explanations | 2 |
| F | Empathy and responsiveness | 2 |
| G | Clarification and summarising | 2 |
| H | Consulting style and organisation | 2 |
| I | Professional behaviour | 2 |
| J | Professional spoken conduct | 2 |

### SPIKES Protocol (Baile et al., 2000)

Six-step bad-news delivery framework annotated in Pass 1 of the LLM analysis:

| Step | Description |
|------|-------------|
| S1 | Setting up |
| P | Patient's perception |
| I | Invitation |
| K | Knowledge delivery |
| E | Empathic response |
| S2 | Strategy and summary |

## Project Structure

```
paed-sim-pipeline/
├── pipeline.py                     # Main orchestrator
├── config/
│   └── pipeline_config.yaml        # All configuration
├── stages/
│   ├── base.py                     # Abstract base stage
│   ├── s1_ingest.py                # Data ingestion & validation
│   ├── s2_asr.py                   # Whisper + speaker diarization
│   ├── s3_features.py              # Verbal feature extraction + phase segmentation
│   ├── s4_video_analysis.py        # Non-verbal behaviour (MediaPipe)
│   ├── s5_analysis.py              # LLM analysis (SPIKES + LUCAS)
│   ├── s6_translate.py             # Translation (disabled by default)
│   └── s7_report.py                # Report generation
├── utils/
│   ├── artifact_io.py              # JSON serialisation helpers
│   ├── freeze.py                   # Freeze manifest
│   ├── json_utils.py               # JSON encoder / sanitiser
│   ├── llm_backends.py             # llama-cpp & vLLM backend abstractions
│   └── logging_setup.py            # Logging config
├── templates/
│   └── clinical_modules/           # Scenario-specific clinical rubric modules
├── download_models.sh              # Model download helper
├── environment.yml                 # Conda environment
└── README.md
```
