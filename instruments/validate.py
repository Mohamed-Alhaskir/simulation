#!/usr/bin/env python3
"""
Validate all instruments/*.json against instruments/schema.json.
Usage: python instruments/validate.py
"""

import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)

INSTRUMENTS_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = INSTRUMENTS_DIR / "schema.json"


def main():
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    instrument_files = sorted(INSTRUMENTS_DIR.glob("*.json"))
    if not instrument_files:
        print(f"No instrument files found in {INSTRUMENTS_DIR}")
        sys.exit(1)

    errors = []
    for path in instrument_files:
        try:
            with open(path) as f:
                instrument = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"  ✗ {path.name}: JSON parse error — {e}")
            continue

        try:
            jsonschema.validate(instrument, schema)
            n_items = len(instrument.get("items", []))
            n_passes = len(instrument.get("passes", []))
            print(f"  ✓ {path.name} ({n_items} items, {n_passes} passes)")
        except jsonschema.ValidationError as e:
            errors.append(f"  ✗ {path.name}: {e.message} (path: {'/'.join(str(p) for p in e.path)})")

    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print(f"\nAll {len(instrument_files)} instrument(s) valid.")


if __name__ == "__main__":
    main()
