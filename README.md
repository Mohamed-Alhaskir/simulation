# Automated Multimodal Feedback Generation for Paediatric Simulation Training: Harmonized Assessment and Joint Feedback

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Pipeline-v0.4.0-green" alt="Pipeline v0.4.0"/>
  <img src="https://img.shields.io/badge/Ethics-S--44%2F2025-orange" alt="Ethics S-44/2025"/>
  <img src="https://img.shields.io/badge/Language-German-lightgrey" alt="Language: German"/>
  <img src="https://img.shields.io/badge/Inference-Local%20LLM-purple" alt="Local LLM"/>
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey" alt="CC BY-NC 4.0"/>
</p>

---

## Abstract

We present a deterministic, reproducible pipeline for automated assessment of communication and clinical competency in paediatric simulation training. The system processes composite audiovisual recordings from standardised patient scenarios and generates structured feedback reports grounded in validated assessment frameworks — the LUCAS communication scale (University of Liverpool, 10 items, max 18 points), the SPIKES bad-news delivery protocol (Baile et al., 2000), and scenario-specific clinical content rubrics (informed-consent structure, clinical quality, and disease-specific knowledge).

The pipeline integrates automatic speech recognition with speaker diarization (Whisper large-v3 + NeMo TitaNet via [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)), non-verbal behaviour analysis (MediaPipe computer vision), and multi-pass large language model inference (Qwen2.5-32B-Instruct, temperature = 0, seed = 42). All model weights, prompts, configuration, and random seeds are cryptographically locked in a **freeze manifest** prior to confirmatory analysis, ensuring full auditability and reproducibility in a clinical research context.

> Study context: Prospective evaluation at the RWTH Aachen University (Medical Informatics). Ethics committee reference: S-44/2025.

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

1. **Multimodal evidence integration** — Communication quality depends on verbal content and non-verbal behaviour simultaneously. The system combines ASR-derived transcripts, verbal interaction metrics, and MediaPipe-extracted gaze/positioning/posture data into a single structured context before any LLM inference.

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
          │  NeMo TitaNet diarization    │
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
          │  Gaze · Positioning · Posture│
          │  Person-relative baselines   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  05 · LLM ANALYSIS           │
          │  LUCAS  (7 passes, all)      │
          │  SPIKES (16 passes, Diabetes)│
          │  Clinical content (per-scen.)│
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  06 · TRANSLATION (optional) │
          │  Disabled by default         │
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

Item D scoring derives from MediaPipe non-verbal behaviour metrics (D1 eye contact via iris tracking, D2 positioning via rolling Hough horizon, D3 posture via arm openness baseline). Items I and J are binary (0 or 2, no borderline score). Item I incorporates video-derived demeanour cues as supplementary evidence alongside the transcript. LLM scoring uses a **7-pass decomposition** rather than a single monolithic prompt, with programmatic validators enforcing metric-based thresholds after each sub-pass. All LLM prompts include anti-hallucination safeguards (QUELLENREGEL: transcript-only evidence, Transkriptbindung: no fabricated timestamps).

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

Scenario-specific checklists evaluating **medical accuracy and completeness**, scored 0/1/2/NA per item. Applied per-scenario via multi-pass LLM calls:

| Scenario | Instruments | Passes | Description |
|----------|-------------|--------|-------------|
| LP\_Aufklaerung | GSLP (structural) | 3 | Beschreibung, Begründung, Sequenzprüfung |
| LP\_Aufklaerung | LP\_Aufklaerung (clinical quality) | 3 | Beschreibung, Begründung, Risiken |
| Diabetes | Diabetes\_CC (clinical content) | 5 | Pathophysiologie, Diagnostik, Therapie, Schuldfrage, Komplikationen |

---

## Installation

### Prerequisites

- NVIDIA GPU with ≥ 34 GB VRAM (for Qwen2.5-32B Q8\_0) or ≥ 42 GB for the 72B variant
- CUDA 12.1
- Conda

### Environment

```bash
conda env create -f environment.yml
conda activate simpipeline
```

### Models

| Component | Model | Source |
|-----------|-------|--------|
| ASR | Whisper large-v3 | Auto-downloaded by `faster-whisper` on first run |
| Diarization | NeMo TitaNet | Via [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) — clone and set `asr.diarization.repo_path` in config |
| LLM | Qwen2.5-32B-Instruct (GGUF Q8\_0) | [Qwen/Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF) — download and set `llm.model_path` in config |

See `config/pipeline_config.example.yaml` for all model-related settings.

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
| `asr` | `model_name` (large-v3), `device` (cuda/cpu), `compute_type`, `beam_size`, `suppress_numerals`, diarization settings (`batch_size`, `temperature`, `no_stem`) |
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
│   ├── video_features.json              # D1–D3 NVB metrics with reliability ratings
│   └── annotated_video.mp4              # Frame-annotated video (optional)
│
├── 05_analysis/
│   ├── assembled_context.json           # Exact LLM input (reproducibility)
│   ├── lucas_pass{1-7}_prompt.txt       # Rendered LUCAS prompts (audit trail)
│   ├── lucas_pass{1-7}_raw_output.txt   # Raw LUCAS LLM responses
│   ├── lucas_analysis.json              # LUCAS items A–J (merged)
│   ├── spikes_pass{1-7}_prompt.txt      # Rendered SPIKES prompts
│   ├── spikes_pass{1-7}_raw_output.txt  # Raw SPIKES LLM responses
│   ├── spikes_annotation.json           # SPIKES phases (merged)
│   ├── {instrument}_prompt.txt          # Clinical content prompts
│   ├── {instrument}_raw_output.txt      # Clinical content LLM responses
│   ├── clinical_content.json            # Clinical checklist (combined)
│   └── analysis.json                    # All instruments merged
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
| `pipeline_version` | Semver string (e.g. `0.4.0`) |
| `git_commit` | SHA of HEAD at freeze time |
| `frozen_at` | UTC ISO timestamp |
| `seeds.global` / `seeds.llm` | Global and LLM seeds (both `42`) |
| `models.asr` | Whisper model, compute type, beam size |
| `models.diarization` | NeMo TitaNet model identifier |
| `models.llm` | Backend, model path, temperature, context length |
| `prompt_template_hashes` | Per-file SHA-256 of all 25 Jinja2 templates + 5 instrument JSON definitions, plus combined digest |
| `config_hash` | SHA-256 of full YAML configuration |
| `manifest_digest` | SHA-256 of the complete manifest |

### Freeze workflow

```bash
# 1. Finalise all code, prompts, and config
# 2. Commit everything
git add -A && git commit -m "Finalise pipeline for confirmatory analysis"

# 3. Generate and commit the manifest
python pipeline.py --config config/pipeline_config.yaml \
                   --freeze-manifest > freeze_manifest.json
git add freeze_manifest.json && git commit -m "Freeze manifest v0.4.0"

# 4. Tag the freeze point
git tag -a v0.4.0-freeze -m "Pipeline freeze for confirmatory analysis"
git push origin main --tags

# 5. Do NOT modify code, prompts, models, or config after this point
```

Every re-run after freeze compares the current state against the archived manifest. Any deviation (code change, prompt edit, config update, different model) raises a `FREEZE VIOLATION` with the specific changed fields before analysis proceeds.

---

## Scenario Routing

The pipeline applies different assessment instruments depending on scenario type, controlled by `_SCENARIO_CONFIG` in [`stages/s5_analysis.py`](stages/s5_analysis.py):

| Scenario | LUCAS | SPIKES | Clinical instruments |
|----------|:-----:|:------:|:-------------------:|
| `Lumbar Puncture` | 7 passes | — | GSLP (3 passes) + LP\_Aufklaerung (3 passes) |
| `Diabetes` | 7 passes | 16 passes | Diabetes\_CC (5 passes) |

New scenarios can be added by:
1. Creating instrument JSON definitions in `instruments/`
2. Creating corresponding Jinja2 templates in `templates/instruments/`
3. Adding an entry to `_SCENARIO_CONFIG` in `stages/s5_analysis.py`
4. Registering sessions in `templates/scenario_catalog.json`

---

## Project Structure

```
paed-sim-pipeline/
│
├── pipeline.py                      # Main orchestrator & CLI
├── config/
│   ├── pipeline_config.yaml         # Single source of truth for all parameters
│   └── pipeline_config.example.yaml # Template config (no secrets/paths)
│
├── stages/
│   ├── base.py                      # Abstract BaseStage
│   ├── s1_ingest.py                 # Video validation, audio extraction, quadrant split
│   ├── s2_asr.py                    # Whisper transcription + NeMo TitaNet diarization
│   ├── s3_features.py               # Verbal interaction features
│   ├── s4_video_analysis.py         # MediaPipe NVB: gaze (D1), positioning (D2), posture (D3)
│   ├── s5_analysis.py               # LLM analysis orchestrator (LUCAS + SPIKES + clinical)
│   ├── s6_translate.py              # Optional translation pass
│   └── s7_report.py                 # HTML/PDF/JSON report generation
│
├── instruments/                     # Assessment instrument definitions
│   ├── LUCAS.json                   # LUCAS 10-item scoring definitions
│   ├── SPIKES.json                  # SPIKES 6-step protocol definitions
│   ├── GSLP.json                    # Informed-consent structure checklist
│   ├── LP_Aufklaerung.json          # LP consent — clinical quality rubric
│   └── Diabetes_CC.json             # T1DM diagnosis disclosure rubric
│
├── templates/
│   ├── scenario_catalog.json        # Session ID → scenario mapping
│   └── instruments/                 # Multi-pass Jinja2 scoring prompts
│       ├── lucas_pass{1-7}.j2       # LUCAS: 7 passes (items → aggregation)
│       ├── spikes_pass{1-7}.j2      # SPIKES: 7 passes (phases → aggregation)
│       ├── gslp_beschreibung.j2     # GSLP: Beschreibung pass
│       ├── gslp_begruendung.j2      # GSLP: Begründung pass
│       ├── gslp_sequence.j2         # GSLP: Sequenzprüfung (aggregation)
│       ├── lp_aufklaerung_*.j2      # LP: Beschreibung, Begründung, Risiken
│       └── diabetes_cc_*.j2         # Diabetes: Pathophysio, Diagnostik, Therapie,
│                                    #           Schuldfrage, Komplikationen
│
├── utils/
│   ├── artifact_io.py               # JSON serialisation helpers
│   ├── freeze.py                    # Freeze manifest generation & verification
│   ├── json_utils.py                # Custom JSON encoder (NumPy, NaN, Inf)
│   ├── llm_backends.py              # llama-cpp-python & vLLM backend abstractions
│   ├── logging_setup.py             # Centralised logging configuration
│   ├── scenario_map.py              # Session → scenario resolution
│   └── scorers/
│       └── instrument_scorer.py     # Template rendering & scoring orchestration
│
├── environment.yml                  # Conda environment specification
└── README.md                        # This file
```

---

## Citation

If you use this pipeline in academic work, please cite:

```bibtex
@software{paed_sim_pipeline_2025,
  title     = {Automated Multimodal Feedback Generation for Paediatric Simulation Training: Harmonized Assessment and Joint Feedback},
  author    = {Mohamed Alhaskir and Hannah Haven and Jonas Bienzeisler},
  year      = {2026},
  version   = {0.4.0},
  institution = {RWTH Aachen University},
  note      = {Ethics ref: S-44/2025}
}
```

**Assessment frameworks:**

- **LUCAS**: Kramer, D., Hillman, T., & Sheringham, J. (2023). Liverpool Undergraduate Communication Assessment Scale (LUCAS): Development and evaluation of a communication skills assessment tool. *Medical Teacher, 45*(10), 1137–1144. https://doi.org/10.1080/0142159X.2023.2197126
- **SPIKES**: Baile, W.F., Buckman, R., Lenzi, R., Glober, G., Beale, E.A., & Kudelka, A.P. (2000). SPIKES — A six-step protocol for delivering bad news: Application to the patient with cancer. *The Oncologist, 5*(4), 302–311. https://doi.org/10.1634/theoncologist.5-4-302
- **GSLP / LP_Aufklaerung (Informed consent structure and clinical quality)**: Bundesministerium der Justiz. Bürgerliches Gesetzbuch §630e — Aufklärungspflichten (Patientenrechtegesetz), 2013. https://www.gesetze-im-internet.de/bgb/__630e.html
- **Diabetes_CC (T1DM clinical content)**: Deutsche Diabetes Gesellschaft (DDG). S3-Leitlinie: Diagnostik, Therapie und Verlaufskontrolle des Diabetes mellitus im Kindes- und Jugendalter. AWMF-Register Nr. 057-016, Version 4.0, 2023. https://register.awmf.org/de/leitlinien/detail/057-016

**Software and models:**

- **Whisper**: Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th ICML*, 28492–28518. https://doi.org/10.48550/arXiv.2212.04356
- **whisper-diarization**: Ashraf, M. (2023). whisper-diarization: Speaker diarization using Whisper and NeMo. https://github.com/MahmoudAshraf97/whisper-diarization
- **NeMo TitaNet**: Koluguri, N.R., Park, T., & Ginsburg, B. (2022). TitaNet: Neural model for speaker representation with 1D Depth-wise separable convolutions and global context. *Proceedings of ICASSP 2022*, 8102–8106. https://doi.org/10.1109/ICASSP43922.2022.9746806
- **MediaPipe**: Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*. https://doi.org/10.48550/arXiv.1906.08172
- **Qwen2.5**: Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., ... & Li, Z. (2024). Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*. https://doi.org/10.48550/arXiv.2412.15115
- **llama.cpp**: Gerganov, G. (2023). llama.cpp: Inference of Meta's LLaMA model in pure C/C++. https://github.com/ggerganov/llama.cpp

---

## License

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). You may share and adapt the material for non-commercial purposes with appropriate attribution.

see [LICENSE](LICENSE) for details.

---

<p align="center">
  RWTH Aachen University · Medical Informatics · Aachen<br/>
</p>
