#!/usr/bin/env python3
"""
Unified comparison: LUCAS + SPIKES + Clinical Content vs Gold Truth

Automatically compares all available assessments for each session:
- Detects which assessment types are available (based on scenario)
- Loads corresponding GT data
- Computes median/IQR consensus
- Generates unified results CSV
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import statistics

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Map scenarios to their clinical modules
SCENARIO_TO_MODULE = {
    "LP_Aufklaerung": "LP_Aufklaerung",
    "GSLP": "GSLP",
    "Diabetes": "Diabetes",
    "Bauchschmerzen": None,  # No clinical content module
}

# SPIKES step names (from GT - matches column names)
SPIKES_STEPS = ["Setting", "Perception", "Invitation", "Knowledge", "Strategy_Summary"]

# Map pipeline phase codes to GT step names
PHASE_TO_GT_STEP = {
    "S": "Setting",
    "P": "Perception",
    "I": "Invitation",
    "K": "Knowledge",
    "E": "Strategy_Summary",  # Exploration during summary
    "S2": "Strategy_Summary",  # Second Setting/Summary
}

# LUCAS items
LUCAS_ITEMS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
LUCAS_ITEM_NAMES = {
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

# ═══════════════════════════════════════════════════════════════════════════
# Load Gold Truth Data
# ═══════════════════════════════════════════════════════════════════════════

def load_lucas_gt():
    """Load LUCAS gold truth"""
    gt = defaultdict(list)
    with open("GT/lucas.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["Video"]
            rater_scores = {
                item: int(row[LUCAS_ITEM_NAMES[item]])
                for item in LUCAS_ITEMS
            }
            gt[video].append(rater_scores)
    return dict(gt)

def load_spikes_gt():
    """Load SPIKES gold truth"""
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
                rater_scores["Total"] = int(row["Total"])
                gt[video].append(rater_scores)
    except FileNotFoundError:
        pass
    return dict(gt)

def load_clinical_gt():
    """Load all clinical content modules"""
    gt = {}
    for module in ["Diabetes", "GSLP", "LP_Aufklaerung"]:
        module_gt = defaultdict(list)
        filepath = f"GT/{module}.csv"
        try:
            with open(filepath) as f:
                # Skip empty lines at start
                lines = f.readlines()
                lines = [l for l in lines if l.strip()]
                reader = csv.DictReader(lines)
                for row in reader:
                    if not row or not row.get("Video"):
                        continue
                    video = row["Video"]
                    # Get all numeric columns (skip Video and Rater)
                    rater_scores = {}
                    for key, val in row.items():
                        if key not in ["Video", "Rater"]:
                            try:
                                rater_scores[key] = int(val)
                            except (ValueError, TypeError):
                                pass
                    if rater_scores:
                        module_gt[video].append(rater_scores)
        except FileNotFoundError:
            pass
        if module_gt:
            gt[module] = dict(module_gt)
    return gt

# ═══════════════════════════════════════════════════════════════════════════
# Get Pipeline Results
# ═══════════════════════════════════════════════════════════════════════════

def get_session_paths():
    """Find all session folders (directories only, not .json files)"""
    return sorted([p for p in Path("data/reports").glob("session_*") if p.is_dir()])

def load_session_predictions(session_path):
    """Load LUCAS, SPIKES, and Clinical predictions from a session"""
    results = {
        "session": session_path.name,
        "scenario": None,
        "lucas": None,
        "spikes": None,
        "clinical": None,
    }

    # Get scenario from metadata
    metadata_path = session_path / "01_ingest" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            scenario = metadata.get("scenario")
            # Scenario can be a dict {"id": "..."} or a string
            if isinstance(scenario, dict):
                results["scenario"] = scenario.get("id")
            else:
                results["scenario"] = scenario

    # Load LUCAS predictions
    lucas_path = session_path / "05_analysis" / "lucas_analysis.json"
    if lucas_path.exists():
        with open(lucas_path) as f:
            lucas_data = json.load(f)
            if "lucas_items" in lucas_data:
                results["lucas"] = {
                    item["item"]: item.get("rating")
                    for item in lucas_data["lucas_items"]
                    if "rating" in item
                }
            elif "items" in lucas_data:
                results["lucas"] = {
                    item["item"]: item.get("score")
                    for item in lucas_data["items"]
                }
            elif "predictions" in lucas_data:
                results["lucas"] = lucas_data["predictions"]

    # Load SPIKES predictions
    spikes_path = session_path / "05_analysis" / "spikes_annotation.json"
    if spikes_path.exists():
        try:
            with open(spikes_path) as f:
                spikes_data = json.load(f)
                if isinstance(spikes_data, dict) and "items" in spikes_data:
                    # Group items by GT step (map phase codes to GT names) and compute average
                    step_scores = defaultdict(list)
                    for item in spikes_data.get("items", []):
                        phase = item.get("phase", "").upper()
                        rating = item.get("rating")
                        if rating is not None and phase:
                            # Map phase code to GT step name
                            gt_step = PHASE_TO_GT_STEP.get(phase, phase)
                            step_scores[gt_step].append(rating)

                    # Average by step and keep only GT step names
                    if step_scores:
                        results["spikes"] = {
                            step: sum(ratings) / len(ratings)
                            for step, ratings in step_scores.items()
                            if step in SPIKES_STEPS  # Only GT-defined steps
                        }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Load Clinical predictions
    clinical_path = session_path / "05_analysis" / "clinical_content.json"
    if clinical_path.exists():
        try:
            with open(clinical_path) as f:
                clinical_data = json.load(f)
                if isinstance(clinical_data, dict) and "items" in clinical_data:
                    # Extract clinical item scores
                    results["clinical"] = {
                        f"{item.get('id', 'unknown')}_{i}": item.get("rating")
                        for i, item in enumerate(clinical_data.get("items", []))
                        if item.get("rating") is not None
                    }
        except (json.JSONDecodeError, IOError):
            pass

    return results

# ═══════════════════════════════════════════════════════════════════════════
# Comparison Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_rater_consensus(rater_scores_list, metric="median"):
    """Compute median/IQR consensus from multiple raters"""
    if not rater_scores_list:
        return None

    # Get all items
    items = set()
    for scores in rater_scores_list:
        items.update(scores.keys())

    consensus = {}
    for item in items:
        values = [s.get(item) for s in rater_scores_list if item in s]
        if values:
            if metric == "median":
                consensus[item] = {
                    "median": statistics.median(values),
                    "q1": sorted(values)[len(values)//4] if len(values) > 1 else values[0],
                    "q3": sorted(values)[3*len(values)//4] if len(values) > 1 else values[0],
                    "n_raters": len(values),
                }
            else:
                consensus[item] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "n_raters": len(values),
                }
    return consensus

def match_video_to_gt(session_name):
    """Find GT video matching this session"""
    # Mapping based on available GT data
    # LP_Aufklaerung sessions (001, 003, 005, 006) have LUCAS + Clinical GT
    # Diabetes sessions (007, 008) have SPIKES + Clinical GT, no LUCAS
    session_videos = {
        "session_001": "2025-01-17_14-25-37-Schockraum-Session 1",  # LUCAS, Clinical
        "session_003": "2025-01-17_15-02-15-Schockraum-Session 1",  # LUCAS, Clinical
        "session_005": "2025-10-10_15-34-21-Schockraum-Session 1",  # LUCAS, GSLP
        "session_006": "2025-10-10_17-30-52-Schockraum-Session 1",  # LUCAS, LP_Aufklaerung
        "session_007": "2025-10-10_16-34-17-Schockraum-Session 1",  # SPIKES, Diabetes
        "session_008": "2025-10-10_14-31-45-Schockraum-Session 1",  # SPIKES, Diabetes
    }
    return session_videos.get(session_name)

def compare_assessment(prediction, gt_consensus, assessment_type):
    """Compare prediction to GT consensus"""
    if not prediction or not gt_consensus:
        return None

    comparison = {
        "in_range": 0,
        "total_items": len(prediction),
        "alignment_percent": 0,
        "details": [],
    }

    for item, pred_score in prediction.items():
        if item not in gt_consensus:
            continue

        gt = gt_consensus[item]
        q1 = gt.get("q1")
        q3 = gt.get("q3")
        median = gt.get("median")

        in_range = q1 <= pred_score <= q3 if q1 and q3 else False
        if in_range:
            comparison["in_range"] += 1

        comparison["details"].append({
            "item": item,
            "predicted": pred_score,
            "median": median,
            "q1": q1,
            "q3": q3,
            "in_range": in_range,
        })

    comparison["alignment_percent"] = (
        100 * comparison["in_range"] / comparison["total_items"]
        if comparison["total_items"] > 0
        else 0
    )
    return comparison

# ═══════════════════════════════════════════════════════════════════════════
# Main Comparison Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_unified_comparison():
    """Run complete comparison across all assessments"""

    print("Loading gold truth data...")
    lucas_gt = load_lucas_gt()
    spikes_gt = load_spikes_gt()
    clinical_gt = load_clinical_gt()

    print(f"  LUCAS: {len(lucas_gt)} videos")
    print(f"  SPIKES: {len(spikes_gt)} videos")
    print(f"  Clinical: {len(clinical_gt)} modules with {sum(len(v) for v in clinical_gt.values())} videos")

    print("\nProcessing sessions...")
    results = []

    for session_path in get_session_paths():
        session = load_session_predictions(session_path)
        gt_video = match_video_to_gt(session["session"])

        row = {
            "session": session["session"],
            "scenario": session["scenario"],
            "gt_video": gt_video or "N/A",
        }

        # ─── LUCAS Comparison ───
        if session["lucas"] and gt_video and gt_video in lucas_gt:
            lucas_consensus = compute_rater_consensus(lucas_gt[gt_video])
            lucas_cmp = compare_assessment(session["lucas"], lucas_consensus, "LUCAS")
            row["lucas_alignment"] = lucas_cmp["alignment_percent"]
            row["lucas_in_range"] = lucas_cmp["in_range"]
            row["lucas_total"] = lucas_cmp["total_items"]
        else:
            row["lucas_alignment"] = None
            row["lucas_in_range"] = None
            row["lucas_total"] = None

        # ─── SPIKES Comparison ───
        if session["spikes"] and gt_video and gt_video in spikes_gt:
            spikes_consensus = compute_rater_consensus(spikes_gt[gt_video])
            spikes_cmp = compare_assessment(session["spikes"], spikes_consensus, "SPIKES")
            row["spikes_alignment"] = spikes_cmp["alignment_percent"]
            row["spikes_in_range"] = spikes_cmp["in_range"]
            row["spikes_total"] = spikes_cmp["total_items"]
        else:
            row["spikes_alignment"] = None
            row["spikes_in_range"] = None
            row["spikes_total"] = None

        # ─── Clinical Comparison ───
        clinical_module = SCENARIO_TO_MODULE.get(session["scenario"])
        if session["clinical"] and clinical_module and clinical_module in clinical_gt:
            if gt_video and gt_video in clinical_gt[clinical_module]:
                clinical_consensus = compute_rater_consensus(clinical_gt[clinical_module][gt_video])
                # Clinical results are nested by module, extract main scores
                if isinstance(session["clinical"], dict):
                    clinical_scores = {}
                    for mod_name, mod_data in session["clinical"].items():
                        if isinstance(mod_data, dict) and "score" in mod_data:
                            clinical_scores[mod_name] = mod_data["score"]
                    if clinical_scores:
                        clinical_cmp = compare_assessment(clinical_scores, clinical_consensus, "Clinical")
                        row["clinical_alignment"] = clinical_cmp["alignment_percent"]
                        row["clinical_in_range"] = clinical_cmp["in_range"]
                        row["clinical_total"] = clinical_cmp["total_items"]

        if "clinical_alignment" not in row:
            row["clinical_alignment"] = None
            row["clinical_in_range"] = None
            row["clinical_total"] = None

        results.append(row)
        print(
            f"  {session['session']:15} "
            f"LUCAS: {row['lucas_alignment'] or 'N/A':>5} | "
            f"SPIKES: {row['spikes_alignment'] or 'N/A':>5} | "
            f"Clinical: {row['clinical_alignment'] or 'N/A':>5}"
        )

    # Save results
    output_path = Path("unified_comparison_results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session", "scenario", "gt_video",
                "lucas_alignment", "lucas_in_range", "lucas_total",
                "spikes_alignment", "spikes_in_range", "spikes_total",
                "clinical_alignment", "clinical_in_range", "clinical_total",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_path}")
    return results

if __name__ == "__main__":
    run_unified_comparison()
