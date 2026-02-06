# Data Directory Specification

This document defines how raw simulation session data must be organized for the pipeline to ingest it correctly.

## Directory Structure

```
data/
├── raw/                              # Raw input data (read-only after transfer)
│   ├── session_001/                  # One directory per simulation session
│   │   ├── metadata.json             # Session metadata (REQUIRED)
│   │   ├── video/                    # Video recordings
│   │   │   ├── camera_main.mp4       # Primary camera
│   │   │   └── camera_side.mp4       # Optional: additional angle
│   │   ├── audio/                    # Separate audio (if not embedded in video)
│   │   │   └── mic_room.wav          # Room microphone
│   │   ├── physio/                   # Physiological data (optional)
│   │   │   └── participant_A.csv     # HR, EDA per participant
│   │   └── eyetracking/             # Eye-tracking data (optional)
│   │       └── participant_A.csv     # Gaze data per participant
│   │
│   ├── session_002/
│   │   ├── metadata.json
│   │   ├── video/
│   │   │   └── recording.mp4
│   │   └── physio/
│   │       └── participant_A.csv
│   │
│   └── ...
│
├── processed/                        # Pipeline intermediate outputs (auto-generated)
│   ├── session_001/
│   │   ├── 01_ingested/
│   │   ├── 02_asr/
│   │   ├── 03_features/
│   │   ├── 04_analysis/
│   │   ├── 05_report/
│   │   └── pipeline_meta.json
│   └── ...
│
└── reports/                          # Final blinded reports (auto-generated)
    ├── REPORT_session_001.json
    ├── REPORT_session_002.json
    └── ...
```

## metadata.json (REQUIRED per session)

Every session directory MUST contain a `metadata.json` file:

```json
{
  "session_id": "session_001",
  "date": "2025-06-15",
  "scenario": {
    "name": "Paediatric Asthma Exacerbation",
    "case_id": "PED-ASTHMA-01",
    "difficulty": "intermediate",
    "learning_objectives": [
      "Structured history taking in acute paediatric presentation",
      "Clear communication with anxious parent",
      "Appropriate clinical reasoning and management plan"
    ]
  },
  "participants": [
    {
      "role": "learner",
      "pseudonym": "PARTICIPANT_A",
      "semester": 8
    },
    {
      "role": "simulated_patient",
      "pseudonym": "SP_001"
    },
    {
      "role": "instructor",
      "pseudonym": "INSTR_001"
    }
  ],
  "recordings": {
    "primary_video": "video/camera_main.mp4",
    "primary_audio": null,
    "additional_video": ["video/camera_side.mp4"]
  },
  "duration_planned_min": 15,
  "site": "Wuppertal",
  "notes": ""
}
```

## File Format Requirements

### Video
| Field | Requirement |
|-------|-------------|
| Formats | `.mp4`, `.avi`, `.mkv`, `.mov` |
| Resolution | Minimum 720p recommended |
| Audio track | Embedded audio is sufficient (no separate audio file needed) |
| Duration | 30s – 60min |

### Audio (only needed if NOT embedded in video)
| Field | Requirement |
|-------|-------------|
| Formats | `.wav` (preferred), `.mp3`, `.flac`, `.m4a` |
| Sample rate | 16 kHz minimum (Whisper optimal) |
| Channels | Mono preferred; stereo accepted |

### Physiological Data
| Field | Requirement |
|-------|-------------|
| Format | `.csv` with headers |
| Required columns | `timestamp`, `hr` (heart rate in bpm) |
| Optional columns | `eda` (electrodermal activity in µS), `hrv_rmssd` |
| Timestamp | Seconds from recording start, or ISO 8601 |

Example:
```csv
timestamp,hr,eda
0.0,72.3,1.24
0.5,73.1,1.26
1.0,71.8,1.31
```

### Eye-Tracking Data
| Field | Requirement |
|-------|-------------|
| Format | `.csv` with headers |
| Required columns | `timestamp`, `x`, `y` |
| Optional columns | `fixation` (0/1), `pupil_diameter` |
| Coordinate system | Normalized (0-1) or pixel-based |

Example:
```csv
timestamp,x,y,fixation
0.0,0.52,0.48,1
0.016,0.52,0.47,1
0.033,0.61,0.33,0
```

## How the Pipeline Accesses Data

```
                    metadata.json
                         │
                         ▼
               ┌─────────────────┐
               │   01 INGEST     │──── Reads metadata.json first
               │                 │──── Scans video/, audio/, physio/, eyetracking/
               │                 │──── Validates formats + durations
               └────────┬────────┘
                         │
            Validated files copied to processed/session_XXX/01_ingested/
                         │
                         ▼
               ┌─────────────────┐
               │   02 ASR        │──── Uses primary_audio from metadata
               │                 │──── OR extracts audio from primary_video
               └────────┬────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   03 FEATURES   │──── Reads transcript from 02_asr/
               │                 │──── Reads physio CSVs from 01_ingested/
               │                 │──── Reads eyetracking CSVs from 01_ingested/
               └────────┬────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   04 ANALYSIS   │──── Reads LLM profile from 03_features/
               │                 │──── Reads metadata.json (scenario context)
               └────────┬────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   05 REPORT     │──── Reads analysis from 04_analysis/
               │                 │──── Writes to reports/ (blinded)
               └─────────────────┘
```

## Naming Conventions

- **Session directories**: `session_001`, `session_002`, ... (zero-padded)
- **Pseudonyms**: `PARTICIPANT_A`, `SP_001`, `INSTR_001` (never real names)
- **Report IDs**: `REPORT_session_001` (generic, blinded)

## Pseudonymization Checklist

Before transferring data to the pipeline server:

- [ ] All filenames contain only pseudonymized identifiers
- [ ] metadata.json uses pseudonyms, not real names
- [ ] Video/audio does NOT have name overlays or identifying text
- [ ] Physiological data files contain no PII in headers or content
- [ ] Transfer log documents who transferred what, when, and how
