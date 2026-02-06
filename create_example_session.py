#!/usr/bin/env python3
"""
Create an example session directory with the correct structure.

Usage:
    python create_example_session.py --session-id session_001
    python create_example_session.py --session-id session_001 --with-physio --with-eyetracking
"""

import argparse
import json
from pathlib import Path


EXAMPLE_METADATA = {
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
        "primary_audio": None,
        "additional_video": []
    },
    "duration_planned_min": 15,
    "site": "Wuppertal",
    "notes": ""
}

EXAMPLE_PHYSIO_CSV = """timestamp,hr,eda
0.0,72.3,1.24
0.5,73.1,1.26
1.0,71.8,1.31
1.5,74.2,1.28
2.0,76.5,1.35
2.5,78.1,1.42
3.0,75.3,1.38
3.5,73.9,1.33
4.0,72.1,1.29
4.5,71.5,1.25
"""

EXAMPLE_EYETRACKING_CSV = """timestamp,x,y,fixation
0.0,0.52,0.48,1
0.016,0.52,0.47,1
0.033,0.53,0.48,1
0.050,0.61,0.33,0
0.066,0.65,0.30,0
0.083,0.70,0.42,1
0.100,0.71,0.43,1
0.116,0.70,0.42,1
0.133,0.55,0.50,0
0.150,0.50,0.52,1
"""


def create_session(base_dir: str, session_id: str, with_physio: bool, with_eyetracking: bool):
    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (session_dir / "video").mkdir(exist_ok=True)
    (session_dir / "audio").mkdir(exist_ok=True)

    # Write metadata
    metadata = {**EXAMPLE_METADATA, "session_id": session_id}
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create placeholder for video (you'll replace this with real recordings)
    readme = session_dir / "video" / "README.txt"
    readme.write_text(
        "Place your simulation video recording here.\n"
        "Expected filename: camera_main.mp4 (or update metadata.json)\n"
    )

    # Physio data
    if with_physio:
        physio_dir = session_dir / "physio"
        physio_dir.mkdir(exist_ok=True)
        (physio_dir / "participant_A.csv").write_text(EXAMPLE_PHYSIO_CSV.strip())

    # Eye-tracking data
    if with_eyetracking:
        et_dir = session_dir / "eyetracking"
        et_dir.mkdir(exist_ok=True)
        (et_dir / "participant_A.csv").write_text(EXAMPLE_EYETRACKING_CSV.strip())

    print(f"✓ Created session directory: {session_dir}")
    print(f"  metadata.json: ✓")
    print(f"  video/:        Place recording here → video/camera_main.mp4")
    print(f"  physio/:       {'✓ (sample data)' if with_physio else '— (skipped)'}")
    print(f"  eyetracking/:  {'✓ (sample data)' if with_eyetracking else '— (skipped)'}")


def main():
    parser = argparse.ArgumentParser(description="Create example session directory")
    parser.add_argument("--session-id", default="session_001")
    parser.add_argument("--base-dir", default="data/raw")
    parser.add_argument("--with-physio", action="store_true", default=True)
    parser.add_argument("--with-eyetracking", action="store_true", default=True)
    args = parser.parse_args()

    create_session(args.base_dir, args.session_id, args.with_physio, args.with_eyetracking)


if __name__ == "__main__":
    main()
