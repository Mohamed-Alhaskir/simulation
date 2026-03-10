# Paediatric Simulation AI Feedback Pipeline — Technical Documentation

> **Study context:** University Witten/Herdecke, Ethics ref: S-44/2025.
> **Pipeline version:** 0.3.0
> **Language:** German (de) — all LLM prompts and output are in German.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Context Object](#3-context-object)
4. [Checkpointing](#4-checkpointing)
5. [Freeze Manifest](#5-freeze-manifest)
6. [Stage 01 — Data Ingestion](#6-stage-01--data-ingestion)
7. [Stage 02 — ASR & Speaker Diarization](#7-stage-02--asr--speaker-diarization)
8. [Stage 03 — Feature Extraction](#8-stage-03--feature-extraction)
9. [Stage 04 — Video Analysis](#9-stage-04--video-analysis)
10. [Stage 05 — LLM Analysis](#10-stage-05--llm-analysis)
11. [Stage 07 — Report Generation](#11-stage-07--report-generation)
12. [Assessment Frameworks](#12-assessment-frameworks)
13. [LLM Backends](#13-llm-backends)
14. [Configuration Reference](#14-configuration-reference)
15. [Running the Pipeline](#15-running-the-pipeline)
16. [Output File Structure](#16-output-file-structure)

---

## 1. Overview

This pipeline processes audiovisual recordings of paediatric simulation scenarios and produces standardised, reproducible feedback reports. It replaces or augments instructor-led debriefings by systematically evaluating:

- **Communication quality** via the LUCAS scale (10 items, max 18 points)
- **Conversation structure** via the SPIKES protocol (6-step bad-news framework)
- **Clinical content accuracy** via a 15-item core checklist plus scenario-specific modules

**Input:** One composite 4-quadrant MP4 per session (2×2 grid, audio embedded).
**Output:** JSON, HTML, and PDF feedback reports per session.

The pipeline is deliberately **linear, deterministic, and single-machine**. There are no external services, no agents, and no dynamic data fetching — all context is assembled in Python before any LLM call.

---

## 2. Architecture & Data Flow

```
INPUT: data/raw/session_XXX/recording.mp4
              │
              ▼
┌─────────────────────────────────┐
│  01_ingest  (DataIngestionStage)│
│  - Validate composite video     │
│  - Extract 16kHz mono WAV       │
│  - Split into 4 quadrant clips  │
│  - Load / auto-generate metadata│
│  - Write inventory.json         │
└─────────────────┬───────────────┘
                  │ ctx["artifacts"]["primary_audio"]
                  ▼
┌─────────────────────────────────┐
│  02_asr  (ASRStage)             │
│  - Trim audio to conversation   │
│  - Whisper large-v3 → segments  │
│  - Pyannote diarization         │
│  - Merge: segment + speaker     │
│  - Write transcript.json/.txt   │
└─────────────────┬───────────────┘
                  │ ctx["artifacts"]["transcript"]
                  ▼
┌─────────────────────────────────┐
│  03_features  (FeatureExtract.) │
│  - Turn-taking, pauses, interr. │
│  - Speaker ratios & latencies   │
│  - Conversation phase segments  │
│  - Monitor OCR (placeholder)    │
│  - Write features.json          │
└──────────┬──────────────────────┘
           │ ctx["artifacts"]["features"]
           │ (includes .phases — required by Stage 04)
           ▼
┌─────────────────────────────────┐
│  04_video_analysis  (VideoAnal.)│
│  - MediaPipe FaceLandmarker     │
│  - MediaPipe PoseLandmarker     │
│  - MediaPipe HandLandmarker     │
│  - Person-relative baselines    │
│  - LUCAS D/I aligned metrics    │
│  - Write video_features.json    │
└──────────┬──────────────────────┘
           │ ctx["artifacts"]["video_features"]
           ▼
┌─────────────────────────────────┐
│  05_analysis  (LLMAnalysisStage)│
│  - Assemble context from 2+3+4  │
│  - Pass 1: SPIKES annotation    │
│  - Pass 2: LUCAS scoring        │
│  - Pass 3: Clinical content     │
│  - Write assembled_context.json │
│    spikes_annotation.json       │
│    lucas_analysis.json          │
│    clinical_content.json        │
└──────────┬──────────────────────┘
           │ ctx["artifacts"]["analysis"]
           ▼
┌─────────────────────────────────┐
│  07_report  (ReportGeneration)  │
│  - Normalise analysis schemas   │
│  - Render JSON / HTML / PDF     │
│  - Write REPORT_session_XXX.*   │
└─────────────────────────────────┘

OUTPUT: data/reports/session_XXX/07_report/REPORT_session_XXX.json/.html/.pdf
```

**Dependency note:** Stage 03 must run before Stage 04 because the video analysis reads `ctx["artifacts"]["features"]["phases"]` to produce per-phase NVB summaries.

---

## 3. Context Object

A single Python `dict` (`ctx`) flows through every stage. It is the only communication mechanism between stages.

```python
ctx = {
    "session_id":   "session_001",          # Derived from input directory name
    "input_path":   "data/raw/session_001", # Original source
    "output_base":  "data/reports/session_001",
    "config":       { ... },                # Full YAML config
    "manifest":     { ... },                # Freeze manifest (see §5)
    "timestamps":   { "ingest": {"status": "OK", "elapsed_s": 12.4}, ... },
    "artifacts":    { ... },                # All stage outputs (see below)
}
```

### Key artifact keys

| Key | Set by | Value |
|-----|--------|-------|
| `metadata` | Stage 01 | Session metadata dict (scenario, participants, site) |
| `inventory` | Stage 01 | Paths and durations for all produced files |
| `primary_audio` | Stage 01 | Path to 16kHz mono WAV |
| `composite_video` | Stage 01 | Path to copied composite MP4 |
| `video_duration_s` | Stage 01 | Float |
| `transcript` | Stage 02 | List of `{speaker, start, end, text, words}` dicts |
| `transcript_path` | Stage 02 | Path to `transcript.json` |
| `features` | Stage 03 | Dict with `verbal`, `phases`, `vitals` |
| `features_path` | Stage 03 | Path to `features.json` |
| `video_features` | Stage 04 | Dict with D1–D5 + summary metrics |
| `video_features_path` | Stage 04 | Path to `video_features.json` |
| `spikes_annotation` | Stage 05 | Parsed SPIKES step dict |
| `lucas_analysis` | Stage 05 | Parsed LUCAS items A–J |
| `clinical_content` | Stage 05 | Parsed clinical checklist results |
| `analysis` | Stage 05 | Combined dict with all three analyses |
| `report_path` | Stage 07 | Path to final JSON report |

**Memory management:** After each stage, large in-memory artifacts (transcript list, features dict, etc.) are replaced in `ctx` with their file path strings. This keeps RAM usage low across the full pipeline. Subsequent stages always reload from disk as the source of truth.

---

## 4. Checkpointing

Each stage writes a `.stage_checkpoint.json` to its output directory on successful completion:

```
data/reports/session_001/01_ingest/.stage_checkpoint.json
data/reports/session_001/02_asr/.stage_checkpoint.json
...
```

**On re-run:** The orchestrator checks for this file before running each stage. If it exists, the stage is skipped and artifacts are loaded from the checkpoint. This allows interrupted pipelines to resume without re-running expensive stages (Whisper, LLM).

**To force a full re-run:** `--force` deletes all checkpoint files before execution.

**Checkpoint content:**
```json
{
  "stage": "asr",
  "saved_at": "2026-03-01T14:23:00Z",
  "artifacts": { ... },
  "timestamp": { "status": "OK", "elapsed_s": 847.2 }
}
```

---

## 5. Freeze Manifest

Before any confirmatory analysis, the pipeline state is cryptographically locked with a **freeze manifest**. This ensures no code, prompt, model weight, or configuration change goes undetected during the study.

**Contents of the manifest:**

| Field | Description |
|-------|-------------|
| `pipeline_version` | Semver string (e.g. `"0.3.0"`) |
| `git_commit` | SHA of HEAD at freeze time |
| `frozen_at` | UTC ISO timestamp |
| `seeds.global` | Global random seed (42) |
| `seeds.llm` | LLM seed (42) |
| `models.asr` | Whisper model name, compute type, beam size |
| `models.diarization` | Pyannote model identifier |
| `models.llm` | Backend, model path, temperature, context length |
| `prompt_template_hash` | SHA-256 of prompt template file |
| `config_hash` | SHA-256 of entire YAML config (sorted) |
| `manifest_digest` | SHA-256 of the entire manifest (16 hex chars) |

**Freeze protocol:**
```bash
python pipeline.py --config config/pipeline_config.yaml --freeze-manifest > freeze_manifest.json
git add freeze_manifest.json && git commit -m "Freeze manifest"
# Do NOT modify code, prompts, models, or config after this point
```

**Verification:** `FreezeManifest.load_and_verify()` compares `pipeline_version`, `config_hash`, `prompt_template_hash`, `models`, and `seeds` between the saved manifest and the current state.

---

## 6. Stage 01 — Data Ingestion

**File:** [stages/s1_ingest.py](stages/s1_ingest.py)
**Output dir:** `01_ingest/`

### What it does

1. **Metadata loading:** Looks for `metadata.json` in the session output directory. If absent, auto-generates one from the scenario catalog (`templates/scenario_catalog.json`) and the directory name. Auto-generated metadata should be reviewed before running.

2. **Video discovery:** Checks `metadata.recordings.composite_video`, then auto-scans the input directory for `.mp4/.avi/.mkv/.mov` files. If multiple videos are found, uses the first (alphabetically) and warns.

3. **Validation:** Uses `ffprobe` to read duration and resolution. Rejects videos outside the configured range (default: 30s – 3600s).

4. **Audio extraction (two-pass loudness normalization):**
   - Pass 1: measures loudness statistics (integrated loudness, LRA, true peak, threshold) via `ffmpeg loudnorm`
   - Pass 2: applies precise normalization using measured values; falls back to single-pass defaults if measurement fails
   - Output: 16kHz mono PCM WAV (`audio_extracted.wav`) with a gentle 70Hz high-pass filter

5. **Quadrant splitting:** Crops the 2×2 composite into four individual MP4 clips using ffmpeg `crop` filters. The resolution is read from ffprobe; defaults to 1920×1080 if unavailable.
   - `quadrant_top_left.mp4` — Overhead camera
   - `quadrant_top_right.mp4` — Patient monitor
   - `quadrant_bottom_left.mp4` — Side camera
   - `quadrant_bottom_right.mp4` — Parent eye-tracking view

6. **Inventory:** Writes `inventory.json` with paths, durations, resolutions, and conversation window (`conversation_start_s`, `conversation_end_s`) for downstream stages.

### Key outputs

```
01_ingest/
├── recording.mp4            # Copy of original
├── audio_extracted.wav      # 16kHz mono, loudness-normalised
├── quadrant_top_left.mp4
├── quadrant_top_right.mp4
├── quadrant_bottom_left.mp4
├── quadrant_bottom_right.mp4
├── metadata.json
├── inventory.json
└── .stage_checkpoint.json
```

---

## 7. Stage 02 — ASR & Speaker Diarization

**File:** [stages/s2_asr.py](stages/s2_asr.py)
**Output dir:** `02_asr/`

### What it does

1. **Audio window:** Reads `conversation_start_s` and `conversation_end_s` from `inventory.json`. Trims the audio to this window before transcription. Timestamps are restored to original timeline after merging.

2. **Whisper transcription** (`faster-whisper`, model: `large-v3`):
   - Language: German (`de`)
   - Beam size: 5
   - Word-level timestamps enabled
   - Auto-detects CUDA/CPU; falls back to CPU + int8 on OOM
   - Returns segments: `{start, end, text, words[]}`

3. **Speaker diarization** (`pyannote/speaker-diarization-3.1`):
   - Requires `HF_TOKEN` environment variable
   - min_speakers: 2, max_speakers: 3
   - Returns speaker turns: `{start, end, speaker}` where speakers are `SPEAKER_00`, `SPEAKER_01`, etc.
   - Skipped silently if HF_TOKEN is not set

4. **Merging:** Each Whisper segment is assigned the speaker whose diarization window has the greatest temporal overlap. Segments with no overlap are labeled `UNKNOWN`.

5. **Timeline restoration:** If audio was trimmed, `conversation_start_s` is added back to all segment timestamps.

6. **Output:**
   - `transcript.json`: list of `{speaker, start, end, text, words}` dicts
   - `transcript.txt`: human-readable format `[MM:SS → MM:SS] SPEAKER_00: text`

### Speaker label convention

The LLM prompts treat `SPEAKER_00` as the **clinician** (the person being evaluated) and `SPEAKER_01`/`SPEAKER_02` as the **patient/parent**. The LUCAS scoring prompt enforces evidence strings for Item I to only cite `SPEAKER_00` turns.

### Cleanup

After the stage completes, the Whisper model is explicitly deleted and `gc.collect()` is called to free GPU/CPU memory before the next stage loads.

---

## 8. Stage 03 — Feature Extraction

**File:** [stages/s3_features.py](stages/s3_features.py)
**Output dir:** `03_features/`

### What it does

Computes structured interaction metrics from the diarized transcript. **Must run before Stage 04** because video analysis reads `features["phases"]`.

#### Verbal / interaction features

Groups consecutive same-speaker transcript segments into **turns**, then computes:

| Metric | Description |
|--------|-------------|
| `turns` | List of `{speaker, start, end, text, duration_s, word_count}` |
| `pauses` | Inter-turn gaps ≥ 2.0s (configurable): timestamp, duration, speakers |
| `interruptions` | Overlapping turns > 0.3s from different speakers |
| `summary.speakers` | Per-speaker: turn count, total duration, word count, speaking ratio, avg turn duration |
| `summary.response_latencies` | Per-speaker: mean gap (in seconds) before they respond, count |

#### Conversation phase segmentation

A **time-based heuristic** (not content-based) divides the conversation into four phases based on relative position within total duration:

| Phase | Relative position |
|-------|-------------------|
| `opening` | 0–10% |
| `main_consultation` | 10–70% |
| `summary_and_plan` | 70–90% |
| `closing` | 90–100% |

Each phase object: `{phase, start_s, end_s, turn_count, duration_s}`.

This segmentation is approximate and should be treated as such in downstream analysis.

#### Patient monitor OCR (placeholder)

If `monitor_ocr.enabled: true` in config, attempts to extract vitals from `quadrant_top_right.mp4`. Currently a **placeholder** — not yet implemented. Returns `null`. Target: Tesseract or a Philips IntelliVue-specialised reader sampling HR, NIBP, SpO2, Temp every 30s.

---

## 9. Stage 04 — Video Analysis

**File:** [stages/s4_video_analysis.py](stages/s4_video_analysis.py)
**Output dir:** `04_video_analysis/`

### What it does

Uses the **MediaPipe Tasks API** (FaceLandmarker, PoseLandmarker, HandLandmarker) to extract non-verbal behaviour (NVB) metrics aligned with the LUCAS assessment scale. Only computes features relevant to LUCAS items D and I.

### Design principles

1. **Person-relative baselines:** All metrics are normalised against each individual's resting values from the first N seconds of video. This removes inter-person appearance differences.

2. **Distribution-based reporting:** Reports continuous distributions (mean, SD, percentiles) rather than binary classifications. Avoids premature discretisation.

3. **Reliability indicators:** Every metric includes a `reliability` field (`"high"/"moderate"/"low"`) so the LLM can weight unreliable measurements appropriately.

4. **Per-phase breakdowns:** NVB metrics are computed per conversation phase (opening / main / summary / closing) using the phase segments from Stage 03.

### Metrics computed

#### D1 — Eye contact
- `gaze_on_target.rate`: proportion of detected frames where gaze is directed at the conversation partner
- `gaze_on_target.distribution`: over time
- Reliability based on face detection rate

#### D2 — Positioning
- `Height_to_patient.mean/std`: normalised Y coordinate of clinician eye level relative to patient — lower values indicate same-level or below positioning (favourable)

#### D3 — Posture / arm openness
- `baseline_arm_deviation.mean/std`: deviation from person's own resting arm position via PoseLandmarker wrist distance

#### D4 — Facial expressions
- `positive_expression_rate.rate`: proportion of frames showing positive/neutral expression using MediaPipe blendshape coefficients (smile, jaw, brow)

#### D5 — Gestures and mannerisms
- `hand_movement_periodicity.is_repetitive`: boolean — whether hand movements show repetitive periodicity (nervousness indicator) via autocorrelation
- `hand_movement_periodicity.periodicity_strength`: strength of detected periodicity

### Video summariser

Before injecting video features into the LLM prompt, `_summarise_video_for_llm()` converts raw metric dicts into concise German prose. This prevents the LLM from misreading raw distributions and producing hallucinated NVB observations. The summariser outputs pre-formatted evidence strings that can be directly copied into Item D evidence fields.

---

## 10. Stage 05 — LLM Analysis

**File:** [stages/s5_analysis.py](stages/s5_analysis.py)
**Output dir:** `05_analysis/`

This is the core analytical stage. It assembles all upstream data into a structured context, then makes three sequential LLM inference calls.

### Context assembly

`_build_context()` combines:
- Diarized transcript (from Stage 02)
- Verbal interaction features and phase data (from Stage 03)
- Video NVB features (from Stage 04, optional)

Everything is serialised to JSON and saved as `assembled_context.json` — the **reproducibility ground truth**. The exact content of this file is what was fed to the LLM.

### Pass 1 — SPIKES Annotation

**Template:** [templates/spikes_prompt.j2](templates/spikes_prompt.j2)
**Output:** `spikes_annotation.json`, `spikes_prompt.txt`, `spikes_raw_output.txt`

The LLM identifies whether each of the six SPIKES steps was performed, provides timestamps, evidence quotes, and sequence correctness.

**Template variables:**
- `{{ transcript }}` — formatted diarized transcript
- `{{ interaction }}` — verbal features JSON
- `{{ conversation_phases }}` — phase segmentation JSON
- `{{ video_nvb_section }}` — pre-interpreted NVB prose (or unavailability note)

**LLM output schema:**
```json
{
  "steps": [
    {
      "id": "S1",
      "name": "Setting up",
      "present": true,
      "start_s": 12.4,
      "end_s": 18.7,
      "evidence": "Kliniker schließt Tür und setzt sich auf Augenhöhe",
      "note": "..."
    },
    ...
  ],
  "sequence_correct": true,
  "sequence_note": "Korrekte Reihenfolge eingehalten",
  "overall_spikes_note": "..."
}
```

**Critical rules enforced in the prompt:**
- `present: true` requires **positive evidence** in transcript or video — absence of counter-evidence is not sufficient
- S1: Self-introduction ≠ Setting up; only active environment preparation counts
- E: Must check every emotional patient utterance and whether the clinician responded empathically (not just factually)

### Pass 2 — LUCAS Scoring

**Template:** [templates/lucas_prompt.j2](templates/lucas_prompt.j2)
**Output:** `lucas_analysis.json`, `lucas_prompt.txt`, `lucas_raw_output.txt`

The LLM scores all 10 LUCAS items (A–J) using the transcript, verbal features, phase data, SPIKES annotation from Pass 1, and NVB metrics.

**Template variables:**
- `{{ transcript }}` — same formatted transcript
- `{{ interaction }}` — verbal features JSON
- `{{ conversation_phases }}` — phase segmentation JSON
- `{{ spikes_annotation }}` — Pass 1 result (provides sequencing evidence)
- `{{ video_nvb_section }}` — pre-interpreted NVB prose

**LLM output schema:**
```json
{
  "lucas_items": [
    {
      "item": "A",
      "name": "Greeting and introduction",
      "score": 1,
      "rating_label": "competent",
      "justification": "...",
      "evidence": ["[00:05] SPEAKER_00: Guten Tag, ich bin Dr. Müller..."],
      "strengths": ["..."],
      "gaps": [],
      "next_steps": []
    },
    ...
  ],
  "total_score": 14,
  "overall_summary": "..."
}
```

**Item-specific rules in the prompt:**
- Item D: NVB video metrics are the **primary and only valid evidence source**; the pre-formatted evidence string from the video summariser must be used verbatim
- Item I: All evidence strings must cite only `SPEAKER_00` turns; no borderline — score is 0 or 2 only
- Item J: Same binary scoring (0 or 2)

### Pass 3 — Clinical Content

**Template:** [templates/clinical_content_prompt.j2](templates/clinical_content_prompt.j2)
**Output:** `clinical_content.json`, `clinical_content_prompt.txt`, `clinical_content_raw_output.txt`

The LLM evaluates the **medical accuracy and completeness** of what was said, separately from communication style.

**Checklist composition:**

**Generic core (15 items, CC01–CC15):** Applied to every scenario.

| ID | Category | Name | Critical |
|----|----------|------|----------|
| CC01 | Situation Appraisal | Acuity recognition & escalation | Yes |
| CC02 | Situation Appraisal | Primary survey logic | No |
| CC03 | Situation Appraisal | Vital sign interpretation | Yes |
| CC04 | Data Gathering | Focused history | No |
| CC05 | Data Gathering | Differential diagnosis | No |
| CC06 | Data Gathering | Relevant investigations | No |
| CC07 | Data Gathering | Allergy & medication safety check | Yes |
| CC08 | Immediate Management | Immediate stabilisation steps | Yes |
| CC09 | Treatment | Evidence-consistent therapy initiation | No |
| CC10 | Medication Safety | Drug / dose / route accuracy | Yes |
| CC11 | Communication | Disposition decision communicated | No |
| CC12 | Follow-up | Reassessment and safety netting plan | No |
| CC13 | Communication | Family communication & consent | No |
| CC14 | Teamwork | Escalation & handover quality | No |
| CC15 | Follow-up | Discharge/follow-up instructions | No |

**Scenario-specific modules:** Loaded from `templates/clinical_modules/<scenario_id>.json` at runtime. Each module can:
- Add scenario-specific items (e.g., CS01–CS08 for LP consent)
- Override core items as `NA` when they do not apply (e.g., CC01–CC03 are `NA` for a consent conversation where the clinician is not managing the acute presentation)

**Scoring scale:**
- `2` = correct AND sufficiently specific for safe care
- `1` = right idea but missing key parameters (timing/dose/threshold) or partially wrong priority
- `0` = missing OR clinically wrong/unsafe (prefix `"FALSCH:"` in justification if actively wrong)
- `NA` = not applicable for this case

**LLM output schema:**
```json
{
  "items": [
    {
      "id": "CC01",
      "score": 2,
      "justification": "Kliniker erkennt kritische Situation korrekt...",
      "evidence": "[02:15] SPEAKER_00: Das Kind ist instabil..."
    }
  ],
  "overall_clinical_note": "..."
}
```

### JSON extraction

After each LLM call, `_extract_json()` uses a regex to find the first valid JSON object or array in the raw output, handling cases where the model outputs preamble text. If extraction fails, the error is logged and the raw output is preserved for debugging.

### Combined analysis output

`analysis.json` merges all three pass results:
```json
{
  "spikes": { ... },
  "lucas": { ... },
  "clinical_content": { ... }
}
```

---

## 11. Stage 07 — Report Generation

**File:** [stages/s7_report.py](stages/s7_report.py)
**Output dir:** `07_report/`

### What it does

Renders the LLM analysis into a standardised, blinded feedback report.

1. **Schema normalisation:** Handles both the German output schema (`lucas_items`/`item`/`rating`) and a fallback English schema (`items`/`id`/`score`) via `_normalise_items()`. Produces a uniform list of:
   ```python
   {
     "item_id": "A",
     "name": "Greeting and introduction",
     "category": "Introductions",
     "max_score": 1,
     "score": 1,
     "justification": "...",
     "evidence": [...],
     "strengths": [...],
     "gaps": [...],
     "next_steps": [...]
   }
   ```

2. **LUCAS section grouping:**
   - **Introductions:** A, B
   - **General:** C, D, E, F, G, H
   - **Professional Behaviour and Conduct:** I, J

3. **Report ID:** Blinded label from config prefix + session ID (e.g., `REPORT_session_001`).

4. **Output formats:**
   - `REPORT_session_001.json` — always generated; primary format
   - `REPORT_session_001.html` — generated if `"html"` in `additional_formats`; uses Jinja2 HTML template
   - `REPORT_session_001.pdf` — generated if `"pdf"` in `additional_formats`; requires `weasyprint`

---

## 12. Assessment Frameworks

### LUCAS Scale (University of Liverpool Communication Assessment Scale)

10 items scored A–J, **maximum total 18 points**.

| Item | Name | Max | Scoring |
|------|------|-----|---------|
| A | Greeting and introduction | 1 | 0/1 (binary) |
| B | Identity check | 1 | 0/1 (binary) |
| C | Audibility and clarity of speech | 2 | 0/1/2 |
| D | Non-verbal behaviour | 2 | 0/1/2 |
| E | Questions, prompts and explanations | 2 | 0/1/2 |
| F | Empathy and responsiveness | 2 | 0/1/2 |
| G | Clarification and summarising | 2 | 0/1/2 |
| H | Consulting style and organisation | 2 | 0/1/2 |
| I | Professional behaviour | 2 | 0/2 only (no borderline) |
| J | Professional spoken conduct | 2 | 0/2 only (no borderline) |

**Evidence sources per item:**

| Item | Transcript | Verbal features | Video NVB | SPIKES annotation |
|------|:----------:|:---------------:|:---------:|:-----------------:|
| A | ✓ | | | |
| B | ✓ | | | |
| C | ✓ | | | |
| D | | | ✓ primary | |
| E | ✓ | ✓ | | |
| F | ✓ | | ✓ | ✓ |
| G | ✓ | | | ✓ |
| H | ✓ | ✓ | | ✓ |
| I | ✓ | | ✓ | |
| J | ✓ | | | |

### SPIKES Protocol (Baile et al., 2000)

Six-step framework for delivering bad news.

| Step | Name | Key evidence requirement |
|------|------|--------------------------|
| S1 | Setting up | Active environment preparation (close door, sit at eye level, eliminate interruptions). Self-introduction alone does NOT count. |
| P | Patient's Perception | Open questions to elicit what patient already knows before delivering news |
| I | Invitation | Explicit or implicit permission to share information |
| K | Knowledge | Warning shot before bad news; plain language; chunking; comprehension checks |
| E | Exploring Emotions / Empathy | Clinician names, validates, and explores emotional responses — a purely factual reply does not qualify |
| S2 | Strategy and Summary | Presents next steps, checks understanding, invites questions, summarises |

**Sequence check:** The prompt verifies whether steps occurred in the correct order (S1 → P → I → K → E → S2) and flags deviations.

---

## 13. LLM Backends

**File:** [utils/llm_backends.py](utils/llm_backends.py)

Two backends are supported, selected via `llm.backend` in config.

### llama_cpp (default)

- Uses `llama-cpp-python` for local CPU/GPU inference
- Loads GGUF model files (e.g., Qwen2.5-32B-Instruct Q8_0)
- Model is lazy-loaded on first call and cached
- Parameters: `n_ctx` (context window), `n_gpu_layers=-1` (all layers on GPU), `seed`
- Cleanup: calls `model.close()` to release memory

### vllm

- Uses `vllm` for optimised batched inference
- Takes model name or path (HuggingFace-compatible)
- Parameters: `max_model_len`, `seed`, `SamplingParams(temperature, top_p, max_tokens)`

### Generation parameters (all backends)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.0 | Deterministic output (greedy decoding) |
| `top_p` | 1.0 | No nucleus sampling |
| `seed` | 42 | Reproducibility |
| `max_tokens` | 10000 | Sufficient for all three passes |
| `context_length` | 32768 | Handles long transcripts + prompts |

---

## 14. Configuration Reference

All settings live in [config/pipeline_config.yaml](config/pipeline_config.yaml). This file is part of the freeze manifest — any change requires a version bump.

### `pipeline`
```yaml
pipeline:
  version: "0.1.0-dev"
  seed: 42
  language: "de"
```

### `paths`
```yaml
paths:
  output_dir: "data/reports"
  scenario_catalog: "templates/scenario_catalog.json"
  clinical_modules_dir: "templates/clinical_modules"
```

### `ingest`
```yaml
ingest:
  accepted_video_formats: [".mp4", ".avi", ".mkv", ".mov"]
  min_duration_s: 30
  max_duration_s: 3600
  composite_video:
    enabled: true
    layout: "2x2"
```

### `asr`
```yaml
asr:
  model_type: "Whisper"
  model_name: "large-v3"
  device: "cuda"
  compute_type: "float16"
  beam_size: 5
  language: "de"
  diarization:
    enabled: true
    model: "pyannote/speaker-diarization-3.1"
    min_speakers: 2
    max_speakers: 3
    hf_token_env: "HF_TOKEN"
```

### `features`
```yaml
features:
  verbal:
    compute_turn_taking: true
    pause_threshold_s: 2.0
    compute_interruptions: true
  monitor_ocr:
    enabled: true      # Placeholder — not yet implemented
```

### `video_analysis`
```yaml
video_analysis:
  enabled: true
  sample_fps: 10       # Frames per second sampled for MediaPipe
```

### `llm`
```yaml
llm:
  backend: "llama_cpp"
  model_path: "/data/.../qwen2.5-32b-instruct-q8_0.gguf"
  context_length: 32768
  temperature: 0.0
  seed: 42
  max_tokens: 10000
  spikes_template: "templates/spikes_prompt.j2"
  lucas_template: "templates/lucas_prompt.j2"
  clinical_content_template: "templates/clinical_content_prompt.j2"
```

### `report`
```yaml
report:
  format: "json"
  additional_formats: ["pdf", "html"]
  template: "templates/report_template.html"
  label_prefix: "REPORT"
```

### `evaluation`
Not part of the generation pipeline — tracked in the manifest for study design reference.
```yaml
evaluation:
  phase1:
    primary_metric: "quadratic_weighted_kappa"
    threshold: 0.80
  phase2:
    primary_metric: "lucas_total_score"
    noninferiority_margin: -15
```

---

## 15. Running the Pipeline

### Prerequisites

```bash
conda env create -f environment.yml
conda activate paed-sim-pipeline
export HF_TOKEN=hf_your_token_here   # Required for pyannote diarization
```

### Single session
```bash
python pipeline.py \
  --config config/pipeline_config.yaml \
  --input data/raw/session_001/
```

### Batch mode (all sessions)
```bash
python pipeline.py \
  --config config/pipeline_config.yaml \
  --input data/raw/ \
  --batch
```

### Force re-run (ignore checkpoints)
```bash
python pipeline.py \
  --config config/pipeline_config.yaml \
  --input data/raw/session_001/ \
  --force
```

### Print freeze manifest
```bash
python pipeline.py \
  --config config/pipeline_config.yaml \
  --freeze-manifest
```

### Log levels
```bash
--log-level DEBUG    # Verbose (includes ctx slimming, prompt rendering)
--log-level INFO     # Default
--log-level WARNING  # Quiet
```

### Input data layout
```
data/raw/
├── session_001/
│   ├── recording.mp4          # Composite 4-quadrant video
│   └── metadata.json          # Optional — auto-generated if absent
├── session_002/
│   └── recording.mp4
```

**`metadata.json` format (optional):**
```json
{
  "session_id": "session_001",
  "date": "2026-01-15",
  "scenario": { "id": "LP_Aufklaerung" },
  "participants": [
    { "role": "learner", "pseudonym": "PARTICIPANT_A" },
    { "role": "simulated_patient", "pseudonym": "SP_001" }
  ],
  "recordings": {
    "composite_video": "recording.mp4"
  },
  "site": "Wuppertal"
}
```

If `metadata.json` is absent, the pipeline auto-generates one from `templates/scenario_catalog.json` (maps `session_id → scenario_id`). Auto-generated metadata uses `PARTICIPANT_A` and `SP_001` as pseudonyms.

---

## 16. Output File Structure

```
data/reports/session_001/
│
├── 01_ingest/
│   ├── recording.mp4                  # Copy of original
│   ├── audio_extracted.wav            # 16kHz mono, loudness-normalised
│   ├── quadrant_top_left.mp4          # Overhead camera
│   ├── quadrant_top_right.mp4         # Patient monitor
│   ├── quadrant_bottom_left.mp4       # Side camera
│   ├── quadrant_bottom_right.mp4      # Parent eye-tracking view
│   ├── metadata.json                  # Session metadata
│   ├── inventory.json                 # Artifact paths + durations
│   └── .stage_checkpoint.json
│
├── 02_asr/
│   ├── audio_for_asr.wav              # Trimmed audio (if windowed)
│   ├── transcript.json                # Diarized transcript with timestamps
│   ├── transcript.txt                 # Human-readable transcript
│   └── .stage_checkpoint.json
│
├── 03_features/
│   ├── features.json                  # Turn-taking, pauses, phases, vitals
│   └── .stage_checkpoint.json
│
├── 04_video_analysis/
│   ├── video_features.json            # D1–D5 NVB metrics with reliability
│   ├── annotated_video.mp4            # Optional frame-annotated video
│   └── .stage_checkpoint.json
│
├── 05_analysis/
│   ├── assembled_context.json         # Ground truth — exact LLM input
│   ├── spikes_prompt.txt              # Pass 1: rendered prompt
│   ├── spikes_raw_output.txt          # Pass 1: raw LLM text
│   ├── spikes_annotation.json         # Pass 1: parsed SPIKES result
│   ├── lucas_prompt.txt               # Pass 2: rendered prompt
│   ├── lucas_raw_output.txt           # Pass 2: raw LLM text
│   ├── lucas_analysis.json            # Pass 2: LUCAS items A–J
│   ├── clinical_content_prompt.txt    # Pass 3: rendered prompt
│   ├── clinical_content_raw_output.txt
│   ├── clinical_content.json          # Pass 3: clinical checklist
│   ├── analysis.json                  # Combined (all three passes)
│   └── .stage_checkpoint.json
│
├── 07_report/
│   ├── REPORT_session_001.json        # Final structured report
│   ├── REPORT_session_001.html        # Printable browser version
│   └── REPORT_session_001.pdf         # PDF (requires weasyprint)
│
└── pipeline_meta.json                 # Run timestamps, stage durations,
                                       # pipeline version, manifest hash
```
