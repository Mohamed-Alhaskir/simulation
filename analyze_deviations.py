#!/usr/bin/env python3
"""
Deviation Analysis: Identify where predictions deviate from GT
and suggest prompt adjustments

Usage:
    python analyze_deviations.py session_005

Shows:
- Items with largest deviations
- Pattern of failures
- Suggested prompt improvements
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

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

def load_lucas_predictions(session_name):
    """Load LUCAS predictions"""
    lucas_path = Path(f"data/reports/{session_name}/05_analysis/lucas_analysis.json")
    if not lucas_path.exists():
        return None

    with open(lucas_path) as f:
        lucas_data = json.load(f)
        if "lucas_items" in lucas_data:
            return {
                item["item"]: {
                    "rating": item.get("rating"),
                    "justification": item.get("justification", "")
                }
                for item in lucas_data["lucas_items"]
            }
    return None

def load_lucas_gt(gt_video):
    """Load LUCAS GT"""
    gt = defaultdict(list)
    with open("GT/lucas.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["Video"]
            if video == gt_video:
                rater_scores = {
                    item: int(row[LUCAS_ITEMS[item]])
                    for item in LUCAS_ITEMS.keys()
                }
                gt[video].append(rater_scores)
    return dict(gt)

def analyze_deviations(predictions, gt_raters):
    """Analyze where predictions deviate from GT"""
    if not gt_raters or not predictions:
        return None

    deviations = []
    for item, pred_data in predictions.items():
        pred_score = pred_data["rating"]
        gt_scores = [r.get(item) for r in gt_raters if item in r]

        if not gt_scores:
            continue

        gt_median = statistics.median(gt_scores)
        deviation = pred_score - gt_median

        deviations.append({
            "item": item,
            "item_name": LUCAS_ITEMS.get(item, "Unknown"),
            "predicted": pred_score,
            "gt_median": gt_median,
            "gt_range": (min(gt_scores), max(gt_scores)),
            "deviation": abs(deviation),
            "direction": "too_high" if deviation > 0 else "too_low",
            "justification": pred_data["justification"]
        })

    # Sort by largest deviations
    return sorted(deviations, key=lambda x: x["deviation"], reverse=True)

def suggest_improvements(deviations):
    """Suggest prompt improvements based on deviations"""
    print("\n" + "="*70)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*70 + "\n")

    # Group by direction
    too_high = [d for d in deviations if d["direction"] == "too_high"]
    too_low = [d for d in deviations if d["direction"] == "too_low"]

    if too_high:
        print("⚠ OVERESTIMATING (scores too high):")
        for d in too_high[:3]:
            print(f"\n  {d['item']}: {d['item_name']}")
            print(f"    Predicted: {d['predicted']}, GT median: {d['gt_median']:.1f}, Deviation: +{d['deviation']:.1f}")
            print(f"    Suggestion: Add more critical/stringent criteria to scoring prompt")
            print(f"    Example: 'Only rate 2 if ALL elements are present and clearly explained'")

    if too_low:
        print("\n⚠ UNDERESTIMATING (scores too low):")
        for d in too_low[:3]:
            print(f"\n  {d['item']}: {d['item_name']}")
            print(f"    Predicted: {d['predicted']}, GT median: {d['gt_median']:.1f}, Deviation: {d['deviation']:.1f}")
            print(f"    Suggestion: Clarify positive indicators in prompt")
            print(f"    Example: 'Rate 1 if element is present even if not perfect'")

    # Pattern analysis
    print("\n" + "="*70)
    print("PATTERN ANALYSIS")
    print("="*70 + "\n")

    avg_deviation = statistics.mean([d["deviation"] for d in deviations])
    print(f"Average deviation: {avg_deviation:.2f} points")
    print(f"Consistent issues: {len(too_high)} items overestimated, {len(too_low)} underestimated")

    if too_high and len(too_high) >= 3:
        print(f"\n→ Systematic bias: Pipeline is too generous")
        print(f"  Action: Increase scoring threshold in prompt")

    if too_low and len(too_low) >= 3:
        print(f"\n→ Systematic bias: Pipeline is too strict")
        print(f"  Action: Lower scoring threshold, add more positive criteria")

def main(session_name):
    print(f"\n{'='*70}")
    print(f"DEVIATION ANALYSIS: {session_name}")
    print(f"{'='*70}\n")

    # Load predictions
    predictions = load_lucas_predictions(session_name)
    if not predictions:
        print(f"❌ No LUCAS predictions for {session_name}")
        return

    # Get GT video
    session_to_video = {
        "session_001": "2025-01-17_14-25-37-Schockraum-Session 1",
        "session_003": "2025-01-17_15-02-15-Schockraum-Session 1",
        "session_005": "2025-10-10_15-34-21-Schockraum-Session 1",
        "session_006": "2025-10-10_17-30-52-Schockraum-Session 1",
    }
    gt_video = session_to_video.get(session_name)
    if not gt_video:
        print(f"⚠ No GT mapping for {session_name}")
        return

    # Load GT
    gt_data = load_lucas_gt(gt_video)
    if not gt_data or gt_video not in gt_data:
        print(f"⚠ No GT data for {gt_video}")
        return

    # Analyze
    deviations = analyze_deviations(predictions, gt_data[gt_video])
    if not deviations:
        print("No deviations to analyze")
        return

    # Show results
    print("DEVIATIONS FROM GT (sorted by magnitude):\n")
    for i, d in enumerate(deviations, 1):
        symbol = "↑" if d["direction"] == "too_high" else "↓"
        print(f"{i}. {d['item']}: {d['item_name']}")
        print(f"   {symbol} Predicted: {d['predicted']}, GT: {d['gt_median']:.1f}, Range: {d['gt_range']}")
        print(f"   Deviation: {d['deviation']:.1f} ({d['direction']})")
        print()

    # Suggestions
    suggest_improvements(deviations)

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_deviations.py <session_name>")
        print("Example: python analyze_deviations.py session_005")
        sys.exit(1)

    main(sys.argv[1])
