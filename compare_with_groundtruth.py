#!/usr/bin/env python3
"""
Simple Comparison: Pipeline Predictions vs Ground Truth

Usage:
    python compare_with_groundtruth.py session_005

Automatically:
    1. Detects scenario type (LP_Aufklaerung, Diabetes, etc.)
    2. Finds GT data for that scenario
    3. Loads pipeline predictions
    4. Compares and shows results
"""

import json
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Map scenarios to GT clinical modules
SCENARIO_GT_MAP = {
    "LP_Aufklaerung": "LP_Aufklaerung.csv",
    "GSLP": "GSLP.csv",
    "Diabetes": "Diabetes.csv",
    "Bauchschmerzen": None,
}

# LUCAS item names (10 items A-J)
LUCAS_ITEMS = {
    'A': 'Greeting and introduction',
    'B': 'Identity check',
    'C': 'Audibility and clarity of speech',
    'D': 'Non-verbal behaviour',
    'E': 'Questions prompts and/or explanations',
    'F': 'Empathy & responsiveness',
    'G': 'Clarification & summarising',
    'H': 'Consulting style & organisation',
    'I': 'Professional behaviour',
    'J': 'Professional spoken/verbal conduct'
}

SPIKES_STEPS = ["Setting", "Perception", "Invitation", "Knowledge", "Strategy_Summary"]

# ═══════════════════════════════════════════════════════════════════════════
# Load Session
# ═══════════════════════════════════════════════════════════════════════════

def load_session_metadata(session_name):
    """Get scenario from session metadata"""
    metadata_path = Path(f"data/reports/{session_name}/01_ingest/metadata.json")
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        meta = json.load(f)
        scenario = meta.get("scenario")
        # Scenario can be dict {"id": "..."} or string
        if isinstance(scenario, dict):
            return scenario.get("id")
        return scenario

def load_predictions(session_name):
    """Load LUCAS, SPIKES, Clinical predictions from session"""
    predictions = {"lucas": None, "spikes": None, "clinical": None}

    # Load LUCAS
    lucas_path = Path(f"data/reports/{session_name}/05_analysis/lucas_analysis.json")
    if lucas_path.exists():
        with open(lucas_path) as f:
            lucas_data = json.load(f)
            if "lucas_items" in lucas_data:
                predictions["lucas"] = {
                    item["item"]: item.get("rating")
                    for item in lucas_data["lucas_items"]
                    if "rating" in item
                }

    # Load SPIKES
    spikes_path = Path(f"data/reports/{session_name}/05_analysis/spikes_annotation.json")
    if spikes_path.exists():
        with open(spikes_path) as f:
            spikes_data = json.load(f)
            if "items" in spikes_data:
                # Group by phase
                phase_scores = defaultdict(list)
                for item in spikes_data["items"]:
                    phase = item.get("phase", "").upper()
                    rating = item.get("rating")
                    if rating is not None and phase:
                        phase_scores[phase].append(rating)

                # Average by phase and map to GT names
                if phase_scores:
                    phase_to_step = {
                        "S": "Setting", "P": "Perception", "I": "Invitation",
                        "K": "Knowledge", "E": "Strategy_Summary", "S2": "Strategy_Summary"
                    }
                    step_scores = defaultdict(list)
                    for phase, ratings in phase_scores.items():
                        step = phase_to_step.get(phase, phase)
                        step_scores[step].extend(ratings)

                    predictions["spikes"] = {
                        step: statistics.mean(ratings)
                        for step, ratings in step_scores.items()
                        if step in SPIKES_STEPS
                    }

    # Load Clinical
    clinical_path = Path(f"data/reports/{session_name}/05_analysis/clinical_content.json")
    if clinical_path.exists():
        with open(clinical_path) as f:
            clinical_data = json.load(f)
            if "items" in clinical_data:
                predictions["clinical"] = {
                    f"{item.get('id', 'item')}_{i}": item.get("rating")
                    for i, item in enumerate(clinical_data["items"])
                    if item.get("rating") is not None
                }

    return predictions

# ═══════════════════════════════════════════════════════════════════════════
# Load Ground Truth
# ═══════════════════════════════════════════════════════════════════════════

def load_gt_lucas():
    """Load LUCAS ground truth"""
    gt = defaultdict(list)
    with open("GT/lucas.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["Video"]
            rater_scores = {
                item: int(row[LUCAS_ITEMS[item]])
                for item in LUCAS_ITEMS.keys()
            }
            gt[video].append(rater_scores)
    return dict(gt)

def load_gt_spikes():
    """Load SPIKES ground truth"""
    gt = defaultdict(list)
    try:
        with open("GT/spikes.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video = row["Video"]
                rater_scores = {
                    step: int(row[step])
                    for step in SPIKES_STEPS
                }
                gt[video].append(rater_scores)
    except FileNotFoundError:
        pass
    return dict(gt)

def load_gt_clinical(gt_file):
    """Load clinical GT for specific module"""
    gt = defaultdict(list)
    try:
        with open(f"GT/{gt_file}") as f:
            lines = [l for l in f.readlines() if l.strip()]
            reader = csv.DictReader(lines)
            for row in reader:
                if not row or not row.get("Video"):
                    continue
                video = row["Video"]
                rater_scores = {}
                for key, val in row.items():
                    if key not in ["Video", "Rater"]:
                        try:
                            rater_scores[key] = int(val)
                        except (ValueError, TypeError):
                            pass
                if rater_scores:
                    gt[video].append(rater_scores)
    except FileNotFoundError:
        pass
    return dict(gt)

def find_gt_video_for_scenario(scenario):
    """Find GT videos available for this scenario"""
    gt_file = SCENARIO_GT_MAP.get(scenario)
    if not gt_file:
        return None

    videos = set()
    try:
        with open(f"GT/{gt_file}") as f:
            lines = [l for l in f.readlines() if l.strip()]
            reader = csv.DictReader(lines)
            for row in reader:
                if row.get("Video"):
                    videos.add(row["Video"])
    except FileNotFoundError:
        pass

    return list(sorted(videos))

# ═══════════════════════════════════════════════════════════════════════════
# Find GT Video
# ═══════════════════════════════════════════════════════════════════════════

def find_gt_video(session_name, scenario, assessment_type):
    """Find matching GT video for this session"""
    # Hardcoded mapping (you can update this based on your data)
    session_to_video = {
        "session_001": "2025-01-17_14-25-37-Schockraum-Session 1",
        "session_003": "2025-01-17_15-02-15-Schockraum-Session 1",
        "session_005": "2025-10-10_15-34-21-Schockraum-Session 1",
        "session_006": "2025-10-10_17-30-52-Schockraum-Session 1",
        "session_007": "2025-10-10_16-34-17-Schockraum-Session 1",
        "session_008": "2025-10-10_14-31-45-Schockraum-Session 1",
    }
    return session_to_video.get(session_name)

# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════

def compute_alignment(predictions, gt_consensus):
    """Compare predictions to GT consensus"""
    if not predictions or not gt_consensus:
        return None

    in_range = 0
    total = len(predictions)
    details = []

    for item, pred_score in predictions.items():
        if item not in gt_consensus:
            continue

        gt = gt_consensus[item]
        q1 = gt.get("q1")
        q3 = gt.get("q3")
        median = gt.get("median")

        in_range_flag = q1 <= pred_score <= q3 if (q1 and q3) else False
        if in_range_flag:
            in_range += 1

        details.append({
            "item": item,
            "predicted": pred_score,
            "median": median,
            "q1": q1,
            "q3": q3,
            "in_range": in_range_flag,
        })

    alignment_percent = (100 * in_range / total) if total > 0 else 0
    return {
        "alignment": alignment_percent,
        "in_range": in_range,
        "total": total,
        "details": details,
    }

def compute_gt_consensus(rater_scores_list):
    """Compute median/IQR from multiple raters"""
    if not rater_scores_list:
        return None

    items = set()
    for scores in rater_scores_list:
        items.update(scores.keys())

    consensus = {}
    for item in items:
        values = [s.get(item) for s in rater_scores_list if item in s]
        if values:
            sorted_vals = sorted(values)
            consensus[item] = {
                "median": statistics.median(values),
                "q1": sorted_vals[len(values)//4],
                "q3": sorted_vals[3*len(values)//4],
                "n_raters": len(values),
            }
    return consensus

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(session_name):
    print(f"\n{'='*70}")
    print(f"COMPARING: {session_name}")
    print(f"{'='*70}\n")

    # 1. Get scenario
    scenario = load_session_metadata(session_name)
    if not scenario:
        print(f"❌ Could not find metadata for {session_name}")
        return
    metadata_path = Path(f"data/reports/{session_name}/01_ingest/metadata.json")
    print(f"✓ Scenario: {scenario}")
    print(f"  Metadata: {metadata_path}")

    # 2. Load predictions
    print(f"\n📂 Loading predictions:")
    predictions = load_predictions(session_name)

    lucas_path = Path(f"data/reports/{session_name}/05_analysis/lucas_analysis.json")
    print(f"  LUCAS: {lucas_path} → {len(predictions['lucas']) if predictions['lucas'] else 0} items")

    spikes_path = Path(f"data/reports/{session_name}/05_analysis/spikes_annotation.json")
    print(f"  SPIKES: {spikes_path} → {len(predictions['spikes']) if predictions['spikes'] else 0} steps")

    clinical_path = Path(f"data/reports/{session_name}/05_analysis/clinical_content.json")
    print(f"  Clinical: {clinical_path} → {len(predictions['clinical']) if predictions['clinical'] else 0} items")

    # 3. Find GT video - first try hardcoded mapping, then search scenario GT files
    gt_video = find_gt_video(session_name, scenario, "lucas")

    # If not found in hardcoded mapping, search the scenario's GT file
    if not gt_video:
        available_videos = find_gt_video_for_scenario(scenario)
        if available_videos:
            # Use first available (or you can prompt user to choose)
            gt_video = available_videos[0]
            print(f"⚠ Using GT video from {scenario}: {gt_video}")
        else:
            print(f"⚠ No GT video found for scenario: {scenario}")
            return
    else:
        print(f"✓ GT video: {gt_video}")

    print(f"\n{'-'*70}")
    print("RESULTS")
    print(f"{'-'*70}\n")

    # 4. Compare LUCAS
    if predictions["lucas"]:
        lucas_gt = load_gt_lucas()
        if gt_video in lucas_gt:
            consensus = compute_gt_consensus(lucas_gt[gt_video])
            result = compute_alignment(predictions["lucas"], consensus)
            if result:
                print(f"📊 LUCAS ALIGNMENT: {result['alignment']:.1f}%")
                print(f"   Items in range: {result['in_range']}/{result['total']}")
                print(f"   Details:")
                for d in result["details"][:5]:  # Show first 5
                    status = "✓" if d["in_range"] else "✗"
                    print(f"     {status} {d['item']}: pred={d['predicted']:.1f}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                print()
        else:
            print(f"⚠ No LUCAS GT data for {gt_video}\n")

    # 5. Compare SPIKES
    if predictions["spikes"]:
        spikes_gt = load_gt_spikes()
        if gt_video in spikes_gt:
            consensus = compute_gt_consensus(spikes_gt[gt_video])
            result = compute_alignment(predictions["spikes"], consensus)
            if result:
                print(f"📊 SPIKES ALIGNMENT: {result['alignment']:.1f}%")
                print(f"   Steps in range: {result['in_range']}/{result['total']}")
                print(f"   Details:")
                for d in result["details"]:
                    status = "✓" if d["in_range"] else "✗"
                    print(f"     {status} {d['item']}: pred={d['predicted']:.1f}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                print()
        else:
            print(f"⚠ No SPIKES GT data for {gt_video}\n")

    # 6. Compare Clinical - search all clinical modules if scenario-specific not found
    if predictions["clinical"]:
        print(f"\n📂 Searching Clinical GT files:")
        clinical_gt = None
        found_module = None

        # First try scenario-specific module
        gt_file = SCENARIO_GT_MAP.get(scenario)
        if gt_file:
            gt_path = f"GT/{gt_file}"
            clinical_gt = load_gt_clinical(gt_file)
            print(f"  [{scenario}] {gt_path} → {len(clinical_gt)} videos")
            if gt_video in clinical_gt:
                found_module = scenario
                print(f"    ✓ Found {gt_video} with {len(clinical_gt[gt_video])} raters")

        # If not found, search all modules
        if not clinical_gt or gt_video not in clinical_gt:
            print(f"  Searching other modules...")
            for module in ["Diabetes", "GSLP", "LP_Aufklaerung"]:
                if module == scenario:
                    continue  # Already checked
                gt_file = f"{module}.csv"
                gt_path = f"GT/{gt_file}"
                clinical_gt = load_gt_clinical(gt_file)
                print(f"  [{module}] {gt_path} → {len(clinical_gt)} videos")
                if gt_video in clinical_gt:
                    found_module = module
                    print(f"    ✓ Found {gt_video} with {len(clinical_gt[gt_video])} raters")
                    break

        if clinical_gt and gt_video in clinical_gt:
            consensus = compute_gt_consensus(clinical_gt[gt_video])
            result = compute_alignment(predictions["clinical"], consensus)
            if result:
                module_label = f" ({found_module})" if found_module != scenario else ""
                print(f"\n📊 CLINICAL ALIGNMENT{module_label}: {result['alignment']:.1f}%")
                print(f"   Items in range: {result['in_range']}/{result['total']}")
                print(f"   GT items: {list(consensus.keys())[:5]}...")  # Show first 5 GT items
                print()
        else:
            print(f"⚠ No Clinical GT data found in any module for {gt_video}\n")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_with_groundtruth.py <session_name>")
        print("Example: python compare_with_groundtruth.py session_005")
        sys.exit(1)

    session = sys.argv[1]
    main(session)
