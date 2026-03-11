# Automated Multimodal Feedback Generation for Paediatric Simulation Training

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Pipeline-v0.3.0-green" alt="Pipeline v0.3.0"/>
  <img src="https://img.shields.io/badge/Ethics-S--44%2F2025-orange" alt="Ethics S-44/2025"/>
  <img src="https://img.shields.io/badge/Language-German-lightgrey" alt="Language: German"/>
  <img src="https://img.shields.io/badge/Inference-Local%20LLM-purple" alt="Local LLM"/>
</p>

---

## Abstract

We present a deterministic, reproducible pipeline for automated assessment of communication and clinical competency in paediatric simulation training. The system processes composite audiovisual recordings from standardised patient scenarios and generates structured feedback reports grounded in validated assessment frameworks — the **LUCAS communication scale** (University of Liverpool, 10 items, max 18 points), the **SPIKES bad-news delivery protocol** (Baile et al., 2000), and **scenario-specific clinical content rubrics** aligned with §630e BGB informed-consent requirements.

The pipeline integrates automatic speech recognition with speaker diarization (Whisper large-v3 + Pyannote), non-verbal behaviour analysis (MediaPipe computer vision), and multi-pass large language model inference (Qwen2.5-32B-Instruct, temperature = 0, seed = 42). All model weights, prompts, configuration, and random seeds are cryptographically locked in a **freeze manifest** prior to confirmatory analysis, ensuring full auditability and reproducibility in a clinical research context.

> **Study context:** Prospective evaluation at the **RWTH Aachen Universtiy** (Medical Informatics). Ethics committee reference: **S-44/2025**. The pipeline augments — but does not replace — instructor-led debriefings.

---

## Table of Contents

1. [Background](#background)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Assessment Frameworks](#assessment-frameworks)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Output Structure](#output-structure)
8. [Reproducibility & Freeze Protocol](#reproducibility--freeze-protocol)
9. [Scenario Routing](#scenario-routing)
10. [Project Structure](#project-structure)
11. [Citation](#citation)

---

## Background

Medical simulation training relies on structured debriefing to translate simulated clinical experiences into learning. Instructor-led debriefings are resource-intensive, subject to inter-rater variability, and limited by observer fatigue in high-throughput programmes. Automated feedback systems offer a scalable complement — provided they are transparent, evidence-grounded, and reproducible enough for research-grade deployment.

This pipeline addresses three core challenges:

1. **Multimodal evidence integration** — Communication quality depends on verbal content, vocal delivery, and non-verbal behaviour simultaneously. The system combines ASR-derived transcripts, verbal interaction metrics, and MediaPipe-extracted gaze/posture/gesture data into a single structured context before any LLM inference.

2. **Assessment framework fidelity** — Rather than holistic LLM judgement, all scoring is grounded in published frameworks with explicit rubrics, mandatory evidence requirements, and programmatic hard-rule validators that override LLM outputs when metric thresholds are violated.

3. **Research-grade reproducibility** — Deterministic inference (temperature = 0, seed = 42), freeze-manifest cryptographic locking, and full intermediate artifact preservation enable exact reconstruction of any analysis from archived inputs.

---

## Pipeline Architecture

```
INPUT: data/raw/session_XXX/recording.mp4
       (composite 4-quadrant video, audio embedded)
                         │
        ┌────────────────▼────────────────┐
        │  FREEZE MANIFEST                 │
        │  git commit · model hashes       │
        │  prompt SHA-256 · config SHA-256 │
        └────────────────┬────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  01 · DATA INGESTION         │
          │  Validate · Extract 16kHz    │
          │  Split 4 quadrant clips      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  02 · ASR & DIARIZATION      │
          │  Whisper large-v3            │
          │  Pyannote 3.1 · LLM relabel  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  03 · FEATURE EXTRACTION     │
          │  Turn-taking · Pauses        │
          │  Phase segmentation          │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  04 · VIDEO ANALYSIS         │
          │  MediaPipe: Face/Pose/Hands  │
          │  Gaze · Posture · Gestures   │
          │  Person-relative baselines   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  05 · LLM ANALYSIS           │
          │  Pass 1: SPIKES annotation   │  ← Diabetes only
          │  Pass 2: LUCAS (7 sub-passes)│  ← All scenarios
          │  Pass 3: Clinical content    │  ← Per module
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  07 · REPORT GENERATION      │
          │  JSON · HTML · PDF           │
          └─────────────────────────────┘

OUTPUT: data/reports/session_XXX/07_report/REPORT_session_XXX.*
```

Each stage writes a `.stage_checkpoint.json` on completion. Re-runs skip completed stages unless `--force` is specified, enabling resumption after interruption without re-running expensive LLM or ASR calls.

---

## Assessment Frameworks

### LUCAS — Liverpool Undergraduate Communication Assessment Scale

Ten items rated A–J, **maximum total 18 points**. Applied to **all scenarios**.

| Item | Domain | Description | Max | Scale |
|------|--------|-------------|-----|-------|
| A | Introductions | Greeting and self-introduction | 1 | 0/1 |
| B | Introductions | Identity check of surrogate | 1 | 0/1 |
| C | General | Audibility and clarity of speech | 2 | 0/1/2 |
| D | General | Non-verbal behaviour | 2 | 0/1/2 |
| E | General | Questions, prompts, and explanations | 2 | 0/1/2 |
| F | General | Empathy and responsiveness | 2 | 0/1/2 |
| G | General | Clarification and summarising | 2 | 0/1/2 |
| H | General | Consulting style and organisation | 2 | 0/1/2 |
| I | Professional | Professional behaviour | 2 | 0/2 only |
| J | Professional | Professional spoken conduct | 2 | 0/2 only |

Scoring for Item D derives exclusively from MediaPipe non-verbal behaviour metrics (gaze rate, arm openness, posture deviation, hand movement periodicity). Items I and J are binary (no borderline score). LLM scoring uses a **7-pass decomposition** (`LucasMultipassScorer`) rather than a single monolithic prompt, with programmatic validators enforcing metric-based thresholds after each sub-pass.

### SPIKES Protocol (Baile et al., 2000)

Six-step framework for bad-news delivery. Applied to **Diabetes diagnosis** scenarios only.

| Step | Name | Key criterion |
|------|------|---------------|
| S1 | Setting up | Active environment preparation (not self-introduction) |
| P | Patient's perception | Open inquiry before information delivery |
| I | Invitation | Explicit or implicit permission obtained |
| K | Knowledge | Warning shot; plain language; chunking |
| E | Empathic response | Named, validated, explored emotion — factual reply insufficient |
| S2 | Strategy and summary | Next steps; comprehension check; questions invited |

### Clinical Content Rubrics

Scenario-specific checklists evaluating **medical accuracy and completeness**, scored 0/1/2/NA per item. Applied per-scenario via separate LLM calls:

| Scenario | Modules | Items |
|----------|---------|-------|
| LP\_Aufklaerung | GSLP (§630e structural) + LP\_Aufklaerung (clinical quality) | 9 + 9 |
| Bauchschmerzen | Bauchschmerzen (history-taking) | 10 |
| Diabetes | Diabetes (T1DM diagnosis disclosure) | 20 |

---

## Installation

### Prerequisites

- NVIDIA GPU with ≥ 34 GB VRAM (for Qwen2.5-32B Q8\_0) or ≥ 42 GB for the 72B variant
- CUDA 12.1
- Conda

### Environment

```bash
conda env create -f environment.yml
conda activate paed-sim-pipeline
```

### Models

**Whisper** — downloaded automatically by `faster-whisper` on first run.

**LLM** — download a GGUF-quantised model:

```bash
# Recommended: Qwen2.5-32B-Instruct Q8_0 (~34 GB VRAM)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \
    qwen2.5-32b-instruct-q8_0-*.gguf --local-dir models/

# Alternative: 72B Q4_XS (~42 GB VRAM, lower quality)
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-GGUF \
    Qwen2.5-72B-Instruct-IQ4_XS.gguf --local-dir models/
```

Update `config/pipeline_config.yaml` → `llm.model_path` accordingly.

**Pyannote** — speaker diarization requires accepting model terms on HuggingFace:

1. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Export your token: `export HF_TOKEN=hf_your_token_here`

---

## Quick Start

### Prepare input data

```
data/raw/
├── session_001/
│   └── recording.mp4      # Composite 4-quadrant video, audio embedded
├── session_002/
│   └── recording.mp4
```

The composite video is a 2×2 grid arranged as:

```
┌─────────────────┬──────────────────┐
│  Overhead cam   │  Patient monitor │
│  (top-left)     │  (top-right)     │
├─────────────────┼──────────────────┤
│  Side camera    │  Parent          │
│  (bottom-left)  │  eye-tracking    │
└─────────────────┴──────────────────┘
```

### Register scenarios

Edit `templates/scenario_catalog.json`:
```json
{
  "session_001": "LP_Aufklaerung",
  "session_002": "Diabetes"
}
```

### Run

```bash
# Single session
python pipeline.py --config config/pipeline_config.yaml \
                   --input data/raw/session_001/

# Batch (all sessions in data/raw/)
python pipeline.py --config config/pipeline_config.yaml \
                   --input data/raw/ --batch

# Force full re-run (ignore stage checkpoints)
python pipeline.py --config config/pipeline_config.yaml \
                   --input data/raw/session_001/ --force

# Generate freeze manifest (before confirmatory analysis)
python pipeline.py --config config/pipeline_config.yaml \
                   --freeze-manifest > freeze_manifest.json
```

---

## Configuration

All parameters live in [`config/pipeline_config.yaml`](config/pipeline_config.yaml). This file is part of the freeze manifest — any change requires a version bump before confirmatory analyses.

| Section | Key parameters |
|---------|---------------|
| `asr` | `model_name` (large-v3), `device` (cuda/cpu), `compute_type`, `beam_size`, diarization on/off, `num_speakers` |
| `llm` | `backend` (llama\_cpp / vllm), `model_path`, `temperature` (0.0), `seed` (42), `context_length`, `repeat_penalty` |
| `video_analysis` | `enabled`, `sample_fps`, detection/tracking confidence thresholds, `calibration_seconds` |
| `features` | `pause_threshold_s`, `compute_interruptions` |
| `report` | `additional_formats` (html/pdf), `label_prefix`, `include_timestamps`, `include_spikes` |

---

## Output Structure

```
data/reports/session_001/
│
├── 01_ingest/
│   ├── audio_extracted.wav              # 16kHz mono, loudness-normalised
│   ├── quadrant_{position}.mp4          # Four individual quadrant clips
│   ├── metadata.json                    # Session metadata
│   └── inventory.json                   # Artifact paths + conversation window
│
├── 02_asr/
│   ├── transcript.json                  # Diarized segments {speaker, start, end, text}
│   └── transcript.txt                   # Human-readable [MM:SS → MM:SS] SPEAKER: text
│
├── 03_features/
│   └── features.json                    # Turn-taking, pauses, phases, response latencies
│
├── 04_video_analysis/
│   ├── video_features.json              # D1–D5 NVB metrics with reliability ratings
│   └── annotated_video.mp4              # Frame-annotated video (optional)
│
├── 05_analysis/
│   ├── assembled_context.json           # Ground truth: exact LLM input (reproducibility)
│   ├── spikes_{pass}_prompt.txt         # Rendered prompt (audit trail)
│   ├── spikes_{pass}_raw_output.txt     # Raw LLM text (audit trail)
│   ├── spikes_annotation.json           # Pass 1: SPIKES step annotation
│   ├── lucas_prompt.txt / raw_output.txt
│   ├── lucas_analysis.json              # Pass 2: LUCAS items A–J
│   ├── clinical_content_{module}_prompt.txt
│   ├── clinical_content_{module}_raw_output.txt
│   ├── clinical_content.json            # Pass 3: clinical checklist (combined)
│   └── analysis.json                    # All three passes merged
│
├── 07_report/
│   ├── REPORT_session_001.json          # Primary structured report
│   ├── REPORT_session_001.html          # Printable browser version
│   └── REPORT_session_001.pdf           # PDF (requires weasyprint)
│
└── pipeline_meta.json                   # Run timestamps, stage durations, manifest hash
```

---

## Reproducibility & Freeze Protocol

The pipeline implements a **freeze manifest** that cryptographically locks all analysis-relevant state before confirmatory data collection:

| Manifest field | Content |
|----------------|---------|
| `pipeline_version` | Semver string (e.g. `0.3.0`) |
| `git_commit` | SHA of HEAD at freeze time |
| `frozen_at` | UTC ISO timestamp |
| `seeds.global` / `seeds.llm` | Global and LLM seeds (both `42`) |
| `models.asr` | Whisper model, compute type, beam size |
| `models.diarization` | Pyannote model identifier |
| `models.llm` | Backend, model path, temperature, context length |
| `prompt_template_hash` | SHA-256 of all Jinja2 prompt templates |
| `config_hash` | SHA-256 of full YAML configuration |
| `manifest_digest` | SHA-256 of the complete manifest |

### Freeze workflow

```bash
# 1. Finalise all code, prompts, and config
# 2. Generate and commit the manifest
python pipeline.py --config config/pipeline_config.yaml \
                   --freeze-manifest > freeze_manifest.json
git add freeze_manifest.json && git commit -m "Freeze manifest v0.3.0"

# 3. Do NOT modify code, prompts, models, or config after this point
# 4. The pipeline verifies manifest integrity on every subsequent run
```

Every re-run after freeze compares the current state against the archived manifest. Any deviation (code change, prompt edit, config update, different model) raises a verification error before analysis proceeds.

---

## Scenario Routing

The pipeline applies different assessment passes depending on scenario type, controlled by `_SCENARIO_CONFIG` in [`stages/s5_analysis.py`](stages/s5_analysis.py):

| Scenario | LUCAS | SPIKES | Clinical modules |
|----------|:-----:|:------:|:---------------:|
| `LP_Aufklaerung` | All sessions | — | GSLP + LP\_Aufklaerung |
| `Diabetes` | All sessions | ✓ | Diabetes |

New scenarios can be added by:
1. Creating a clinical module JSON in `templates/clinical_modules/`
2. Adding an entry to `_SCENARIO_CONFIG` in `stages/s5_analysis.py`
3. Registering sessions in `templates/scenario_catalog.json`

---

## Project Structure

```
paed-sim-pipeline/
│
├── pipeline.py                      # Main orchestrator & CLI
├── config/
│   └── pipeline_config.yaml         # Single source of truth for all parameters
│
├── stages/
│   ├── base.py                      # Abstract BaseStage
│   ├── s1_ingest.py                 # Video validation, audio extraction, quadrant split
│   ├── s2_asr.py                    # Whisper transcription + Pyannote diarization
│   ├── s3_features.py               # Verbal interaction features + phase segmentation
│   ├── s4_video_analysis.py         # MediaPipe NVB: gaze, posture, gestures
│   ├── s5_analysis.py               # LLM analysis orchestrator (SPIKES + LUCAS + clinical)
│   ├── s6_translate.py              # Optional translation pass (disabled by default)
│   ├── s7_report.py                 # HTML/PDF/JSON report generation
│   └── lucas_multipass.py           # 7-pass LUCAS scorer with hard-rule validators
│
├── utils/
│   ├── artifact_io.py               # JSON serialisation helpers
│   ├── freeze.py                    # Freeze manifest generation & verification
│   ├── json_utils.py                # Custom JSON encoder (NumPy, NaN, Inf)
│   ├── llm_backends.py              # llama-cpp-python & vLLM backend abstractions
│   ├── logging_setup.py             # Centralised logging configuration
│   └── scenario_map.py              # Session → scenario resolution
│
├── templates/
│   ├── lucas_prompt.j2              # LUCAS 10-item scoring prompt (Jinja2)
│   ├── spikes_prompt.j2             # SPIKES 6-step annotation prompt
│   ├── clinical_content_prompt.j2   # Clinical content evaluation prompt
│   ├── scenario_catalog.json        # Session ID → scenario ID mapping
│   └── clinical_modules/
│       ├── GSLP.json                # §630e BGB consent structure checklist
│       ├── LP_Aufklaerung.json      # LP consent — clinical quality rubric
│       ├── Bauchschmerzen.json      # Abdominal pain history-taking rubric
│       └── Diabetes.json            # T1DM diagnosis disclosure rubric
│
├── tasks/
│   ├── todo.md                      # Active task tracking
│   └── lessons.md                   # Development learnings log
│
├── environment.yml                  # Conda environment specification
├── PIPELINE.md                      # Full technical documentation
├── DATA_SPEC.md                     # Input/output file specification
└── README.md                        # This file
```

---

## Citation

If you use this pipeline in academic work, please cite:

```bibtex
@software{paed_sim_pipeline_2025,
  title     = {Automated Multimodal Feedback Generation for Paediatric Simulation Training},
  author    = {[Mohamed Alhaskir, Hannah Haven, Jonas Bienzeisler]},
  year      = {2026},
  version   = {0.3.0},
  institution = {RWTH Aachen University},
  note      = {Ethics ref: S-44/2025}
}
```

**Assessment frameworks:**

- LUCAS: [University of Liverpool Communication Assessment Scale — contact UoL for licensing]
- SPIKES: Baile, W.F., Buckman, R., Lenzi, R., Glober, G., Beale, E.A., & Kudelka, A.P. (2000). SPIKES — A six-step protocol for delivering bad news. *The Oncologist, 5*(4), 302–311. https://doi.org/10.1634/theoncologist.5-4-302
- Informed consent framework: §630e BGB (Patientenrechtegesetz, Germany)

---

## License

This repository is made available for **research and academic use** under the terms described in [LICENSE](LICENSE). The clinical assessment rubrics (LUCAS, SPIKES, clinical modules) are reproduced for research purposes only and remain subject to their respective source licences.

---

<p align="center">
  RWTH Aachen University· Medical Informatics · Aachen<br/>
</p>
