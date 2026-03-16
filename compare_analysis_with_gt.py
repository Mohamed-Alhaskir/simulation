#!/usr/bin/env python3
"""
Direct comparison: analysis.json predictions vs Ground Truth

Loads directly from analysis.json (instead of separate files)
Handles clinical item ID to GT column mapping
"""

import json
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

# Clinical item ID to GT column mapping
CLINICAL_MAPPING = {
    # GSLP
    "LP_GS_1": "Type_of_treatment",
    "LP_GS_2": "Scope",
    "LP_GS_3": "Procedure",
    "LP_GS_4": "Consequences",
    "LP_GS_5": "Risks",
    "LP_GS_6": "Necessity_Urgency",
    "LP_GS_7": "Suitability",
    "LP_GS_8": "Chances_of_success",
    "LP_GS_9": "Alternatives",
    # LP_Aufklaerung
    "LP_A": "Art",
    "LP_B": "Umfang",
    "LP_C": "Durchfuehrung",
    "LP_D": "Moegliche_Folgen",
    "LP_E": "Risiken",
    "LP_F": "Notwendigkeit_Dringlichkeit",
    "LP_G": "Eignung",
    "LP_H": "Erfolgsaussichten",
    "LP_I": "Behandlungsalternativen",
}

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

# SPIKES item ID to GT column mapping (by position in GT file)
SPIKES_MAPPING = {
    "DM_S_1": "Structured greeting",
    "DM_S_2": "Introduces role and reason for visit",
    "DM_S_3": "Ensures patient comfort",
    "DM_S_4": "Maintains eye contact and attentive body language",
    "DM_S_5": "Manages interruptions appropriately",
    "DM_P_6": "Checks patient's perception of situation",
    "DM_P_7": "Uses understandable language",
    "DM_P_8": "Corrects misinformation sensitively",
    "DM_I_9": "Uses majority of open questions",
    "DM_K_10": "Provides sufficient information",
    "DM_K_11": "Avoids unnecessary technical jargon",
    "DM_K_12": "Gives clear next steps / instructions",
    "DM_K_13": "Allows pauses for patient response",
    "DM_K_14": "Provides hopeful but realistic outlook",
    "DM_E_15": "Recognizes patient emotions",
    "DM_E_16": "Responds empathetically to emotions",
    "DM_E_17": "Checks understanding of emotions",
    "DM_E_18": "Uses calm and supportive tone",
    "DM_E_19": "Ensures patient feels heard",
    "DM_E_20": "Summarizes conversation",
    "DM_S_21": "Shared decision-making",
    "DM_S_22": "Checks for misunderstandings",
    "DM_S_23": "Encourages patient questions",
    "DM_S_24": "Uses clear and structured closing",
}

# Diabetes clinical item ID to GT column mapping
DIABETES_MAPPING = {
    "DM_A_1": "Verdeutlicht dass es sich um Typ-1-Diabetes handelt und es nicht dieselbe Erkrankung wie Typ-2 ist",
    "DM_A_2": "Erklärt grundsätzliche Unterschiede zwischen Typ-1 und Typ-2 Diabetes",
    "DM_A_5": "Erklärt was Insulin ist wenn die Bezugsperson danach fragt",
    "DM_B_6": "Nennt typische Symptome als Begründung für die Diagnose",
    "DM_B_7": "Nennt pathologische Blutzuckermessung als Grundlage der Diagnose",
    "DM_C_8": "Gibt korrekte epidemiologische Informationen zum Typ-1-Diabetes",
    "DM_D_9": "Erklärt dass Typ-1-Diabetes eine chronische lebenslang insulinpflichtige Erkrankung ist",
    "DM_D_10": "Erklärt die empfohlene Insulintherapie im Alltag",
    "DM_D_11": "Erwähnt technologische Hilfsmittel (Insulinpumpe CGM AID-Systeme)",
    "DM_D_12": "Erklärt den Umgang mit Ernährung bei Typ-1-Diabetes",
    "DM_D_13": "Erklärt den Umgang mit Sport bei Typ-1-Diabetes",
    "DM_D_14": "Informiert über Schulungsprogramme und psychosoziale Unterstützung",
    "DM_E_15": "Nennt mindestens 4 Langzeitkomplikationen des Typ-1-Diabetes",
    "DM_E_19": "Beruhigt die Mutter mit sachlich korrekter Aussage über Prävention der Spätfolgen",
    "DM_S_20": "Erklärt dass Typ-1-Diabetes nicht vermeidbar ist und die Eltern keine Schuld haben",
}

# ═══════════════════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════════════════

def load_analysis(session_name):
    """Load predictions from analysis.json"""
    analysis_path = Path(f"data/reports/{session_name}/05_analysis/analysis.json")
    if not analysis_path.exists():
        return None

    with open(analysis_path) as f:
        return json.load(f)

def load_gt_for_video(gt_file, gt_video):
    """Load GT data for specific video"""
    gt = []
    try:
        with open(f"GT/{gt_file}") as f:
            lines = [l for l in f.readlines() if l.strip()]
            reader = csv.DictReader(lines)
            for row in reader:
                if row.get("Video") == gt_video:
                    gt.append(row)
    except FileNotFoundError:
        pass
    return gt

# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════

def compute_alignment(predictions, gt_raters):
    """Compare predictions to GT consensus (only for mapped items)"""
    if not predictions or not gt_raters:
        return None

    # Only process items that have a GT mapping
    mappable_items = {item_id: pred for item_id, pred in predictions.items()
                      if item_id in CLINICAL_MAPPING}

    if not mappable_items:
        return None

    in_range = 0
    total = len(mappable_items)
    details = []

    for item_id, pred_score in mappable_items.items():
        gt_col = CLINICAL_MAPPING[item_id]

        # Find corresponding GT values for this item
        gt_values = []
        for gt_row in gt_raters:
            if gt_col in gt_row:
                try:
                    gt_values.append(int(gt_row[gt_col]))
                except (ValueError, TypeError):
                    pass

        if not gt_values:
            continue

        # Compute consensus
        gt_median = statistics.median(gt_values)
        sorted_vals = sorted(gt_values)
        q1 = sorted_vals[len(gt_values)//4]
        q3 = sorted_vals[3*len(gt_values)//4]

        in_range_flag = q1 <= pred_score <= q3
        if in_range_flag:
            in_range += 1

        details.append({
            "item": item_id,
            "gt_column": gt_col,
            "predicted": pred_score,
            "median": gt_median,
            "q1": q1,
            "q3": q3,
            "in_range": in_range_flag,
        })

    alignment_percent = (100 * in_range / total) if total > 0 else 0
    return {
        "alignment": alignment_percent,
        "in_range": in_range,
        "total": total,
        "mappable_items": len(mappable_items),
        "total_items": len(predictions),
        "details": details,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(session_name):
    print(f"\n{'='*70}")
    print(f"COMPARING: {session_name}")
    print(f"{'='*70}\n")

    # Load analysis
    analysis = load_analysis(session_name)
    if not analysis:
        print(f"❌ Could not find analysis.json for {session_name}")
        return

    scenario = analysis.get("scenario_id")
    print(f"✓ Scenario: {scenario}")
    print(f"📂 File: data/reports/{session_name}/05_analysis/analysis.json\n")

    # Extract predictions
    lucas_pred = {}
    if "lucas_analysis" in analysis:
        for item in analysis["lucas_analysis"].get("lucas_items", []):
            lucas_pred[item["item"]] = item["rating"]

    clinical_pred = {}
    if "clinical_content" in analysis:
        for item in analysis["clinical_content"].get("items", []):
            rating = item["rating"]
            # Skip items with 'NA' or non-numeric ratings
            if rating != "NA":
                try:
                    clinical_pred[item["id"]] = int(rating) if isinstance(rating, str) else rating
                except (ValueError, TypeError):
                    pass

    spikes_pred = {}
    if "spikes_annotation" in analysis:
        for item in analysis["spikes_annotation"].get("items", []):
            rating = item["rating"]
            # Skip items with 'NA' or non-numeric ratings
            if rating != "NA":
                try:
                    spikes_pred[item["id"]] = int(rating) if isinstance(rating, str) else rating
                except (ValueError, TypeError):
                    pass

    print(f"Predictions loaded:")
    print(f"  - LUCAS: {len(lucas_pred)} items")
    print(f"  - SPIKES: {len(spikes_pred)} items")
    print(f"  - Clinical: {len(clinical_pred)} items\n")

    # Session to GT video mapping
    session_to_video = {
        "session_001": "2025-01-17_14-25-37-Schockraum-Session 1",
        "session_003": "2025-01-17_15-02-15-Schockraum-Session 1",
        "session_005": "2025-10-10_15-34-21-Schockraum-Session 1",
        "session_006": "2025-10-10_17-30-52-Schockraum-Session 1",
        "session_007": "2025-10-10_16-34-17-Schockraum-Session 1",
        "session_008": "2025-10-10_14-31-45-Schockraum-Session 1",
    }
    gt_video = session_to_video.get(session_name)

    if not gt_video:
        print(f"⚠ No GT video mapping for {session_name}")
        return

    print(f"GT video: {gt_video}\n")
    print(f"{'-'*70}")
    print("RESULTS")
    print(f"{'-'*70}\n")

    # Compare LUCAS
    if lucas_pred:
        lucas_gt = load_gt_for_video("lucas.csv", gt_video)
        if lucas_gt:
            result = compute_alignment(lucas_pred, lucas_gt)
            if result:
                print(f"📊 LUCAS ALIGNMENT: {result['alignment']:.1f}%")
                print(f"   Items in range: {result['in_range']}/{result['total']}")
                print(f"   Details:")
                for d in result["details"][:5]:
                    status = "✓" if d["in_range"] else "✗"
                    print(f"     {status} {d['item']}: pred={d['predicted']}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                print()

    # Compare Clinical - may have multiple modules per scenario
    if clinical_pred:
        # Separate predictions by module prefix
        gslp_pred = {k: v for k, v in clinical_pred.items() if k.startswith("LP_GS_")}
        lp_auf_pred = {k: v for k, v in clinical_pred.items() if k.startswith("LP_") and not k.startswith("LP_GS_")}

        # Compare each module if it has predictions
        results_shown = False

        if gslp_pred:
            gt_data = load_gt_for_video("GSLP.csv", gt_video)
            if gt_data:
                result = compute_alignment(gslp_pred, gt_data)
                if result:
                    print(f"📊 CLINICAL ALIGNMENT (GSLP): {result['alignment']:.1f}%")
                    print(f"   Mappable items: {result['mappable_items']}/{result['total_items']}")
                    print(f"   Items in range: {result['in_range']}/{result['total']}")
                    print(f"   GT file: GT/GSLP.csv")
                    print(f"   Details:")
                    for d in result["details"]:
                        status = "✓" if d["in_range"] else "✗"
                        print(f"     {status} {d['item']} → {d['gt_column']}: pred={d['predicted']}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                    print()
                    results_shown = True

        if lp_auf_pred:
            gt_data = load_gt_for_video("LP_Aufklaerung.csv", gt_video)
            if gt_data:
                result = compute_alignment(lp_auf_pred, gt_data)
                if result:
                    print(f"📊 CLINICAL ALIGNMENT (LP_Aufklaerung): {result['alignment']:.1f}%")
                    print(f"   Mappable items: {result['mappable_items']}/{result['total_items']}")
                    print(f"   Items in range: {result['in_range']}/{result['total']}")
                    print(f"   GT file: GT/LP_Aufklaerung.csv")
                    print(f"   Details:")
                    for d in result["details"]:
                        status = "✓" if d["in_range"] else "✗"
                        print(f"     {status} {d['item']} → {d['gt_column']}: pred={d['predicted']}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                    print()
                    results_shown = True

        if not results_shown:
            print(f"⚠ No Clinical GT data found for {gt_video}\n")

    # Compare SPIKES
    if spikes_pred:
        spikes_gt = load_gt_for_video("spikes.csv", gt_video)
        if spikes_gt:
            # Manually compute alignment for SPIKES using SPIKES_MAPPING
            mappable_items = {item_id: pred for item_id, pred in spikes_pred.items()
                            if item_id in SPIKES_MAPPING}

            if mappable_items:
                in_range = 0
                total = len(mappable_items)
                details = []

                for item_id, pred_score in mappable_items.items():
                    gt_col = SPIKES_MAPPING[item_id]

                    # Find corresponding GT values for this item
                    gt_values = []
                    for gt_row in spikes_gt:
                        if gt_col in gt_row:
                            try:
                                gt_values.append(int(gt_row[gt_col]))
                            except (ValueError, TypeError):
                                pass

                    if not gt_values:
                        continue

                    # Compute consensus
                    gt_median = statistics.median(gt_values)
                    sorted_vals = sorted(gt_values)
                    q1 = sorted_vals[len(gt_values)//4]
                    q3 = sorted_vals[3*len(gt_values)//4]

                    in_range_flag = q1 <= pred_score <= q3
                    if in_range_flag:
                        in_range += 1

                    details.append({
                        "item": item_id,
                        "gt_column": gt_col,
                        "predicted": pred_score,
                        "median": gt_median,
                        "q1": q1,
                        "q3": q3,
                        "in_range": in_range_flag,
                    })

                alignment_pct = (100 * in_range / total) if total > 0 else 0
                print(f"📊 SPIKES ALIGNMENT: {alignment_pct:.1f}%")
                print(f"   Items in range: {in_range}/{total}")
                print(f"   GT file: GT/spikes.csv")
                print(f"   Details:")
                for d in details:
                    status = "✓" if d["in_range"] else "✗"
                    print(f"     {status} {d['item']} → {d['gt_column']}: pred={d['predicted']}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                print()

    # Compare Diabetes Clinical
    diabetes_pred = {k: v for k, v in clinical_pred.items() if k.startswith("DM_")}
    if diabetes_pred:
        diabetes_gt = load_gt_for_video("Diabetes.csv", gt_video)
        if diabetes_gt:
            # Manually compute alignment for Diabetes using DIABETES_MAPPING
            mappable_items = {item_id: pred for item_id, pred in diabetes_pred.items()
                            if item_id in DIABETES_MAPPING}

            if mappable_items:
                in_range = 0
                total = len(mappable_items)
                details = []

                for item_id, pred_score in mappable_items.items():
                    gt_col = DIABETES_MAPPING[item_id]

                    # Find corresponding GT values for this item
                    gt_values = []
                    for gt_row in diabetes_gt:
                        if gt_col in gt_row:
                            try:
                                gt_values.append(int(gt_row[gt_col]))
                            except (ValueError, TypeError):
                                pass

                    if not gt_values:
                        continue

                    # Compute consensus
                    gt_median = statistics.median(gt_values)
                    sorted_vals = sorted(gt_values)
                    q1 = sorted_vals[len(gt_values)//4]
                    q3 = sorted_vals[3*len(gt_values)//4]

                    in_range_flag = q1 <= pred_score <= q3
                    if in_range_flag:
                        in_range += 1

                    details.append({
                        "item": item_id,
                        "gt_column": gt_col,
                        "predicted": pred_score,
                        "median": gt_median,
                        "q1": q1,
                        "q3": q3,
                        "in_range": in_range_flag,
                    })

                alignment_pct = (100 * in_range / total) if total > 0 else 0
                print(f"📊 DIABETES CLINICAL ALIGNMENT: {alignment_pct:.1f}%")
                print(f"   Items in range: {in_range}/{total}")
                print(f"   GT file: GT/Diabetes.csv")
                print(f"   Details:")
                for d in details:
                    status = "✓" if d["in_range"] else "✗"
                    print(f"     {status} {d['item']} → {d['gt_column'][:50]}: pred={d['predicted']}, median={d['median']:.1f}, IQR=[{d['q1']:.1f}, {d['q3']:.1f}]")
                print()

    print(f"{'='*70}\n")

def save_results_csv(session_name, output_file):
    """Save all comparison results to CSV file"""
    # Load analysis
    analysis = load_analysis(session_name)
    if not analysis:
        print(f"❌ Could not find analysis.json for {session_name}")
        return

    scenario = analysis.get("scenario_id")
    print(f"✓ Scenario: {scenario}")
    print(f"📂 File: data/reports/{session_name}/05_analysis/analysis.json\n")

    # Extract predictions
    lucas_pred = {}
    if "lucas_analysis" in analysis:
        for item in analysis["lucas_analysis"].get("lucas_items", []):
            lucas_pred[item["item"]] = item["rating"]

    clinical_pred = {}
    if "clinical_content" in analysis:
        for item in analysis["clinical_content"].get("items", []):
            rating = item["rating"]
            # Skip items with 'NA' or non-numeric ratings
            if rating != "NA":
                try:
                    clinical_pred[item["id"]] = int(rating) if isinstance(rating, str) else rating
                except (ValueError, TypeError):
                    pass

    spikes_pred = {}
    if "spikes_annotation" in analysis:
        for item in analysis["spikes_annotation"].get("items", []):
            rating = item["rating"]
            # Skip items with 'NA' or non-numeric ratings
            if rating != "NA":
                try:
                    spikes_pred[item["id"]] = int(rating) if isinstance(rating, str) else rating
                except (ValueError, TypeError):
                    pass

    # Session to GT video mapping
    session_to_video = {
        "session_001": "2025-01-17_14-25-37-Schockraum-Session 1",
        "session_003": "2025-01-17_15-02-15-Schockraum-Session 1",
        "session_005": "2025-10-10_15-34-21-Schockraum-Session 1",
        "session_006": "2025-10-10_17-30-52-Schockraum-Session 1",
        "session_007": "2025-10-10_16-34-17-Schockraum-Session 1",
        "session_008": "2025-10-10_14-31-45-Schockraum-Session 1",
    }
    gt_video = session_to_video.get(session_name)

    if not gt_video:
        print(f"⚠ No GT video mapping for {session_name}")
        return

    # Collect all results
    rows = []

    # LUCAS results - compare all 10 items directly
    if lucas_pred:
        lucas_gt = load_gt_for_video("lucas.csv", gt_video)
        if lucas_gt:
            # Convert raw GT rows to rater score dicts
            rater_scores_list = []
            for gt_row in lucas_gt:
                scores = {}
                for item_key in LUCAS_ITEMS.keys():
                    item_name = LUCAS_ITEMS[item_key]
                    try:
                        scores[item_key] = int(gt_row.get(item_name, 0))
                    except (ValueError, TypeError):
                        pass
                if scores:
                    rater_scores_list.append(scores)

            if rater_scores_list:
                # Compute consensus for each item
                for item_key in sorted(lucas_pred.keys()):
                    gt_values = [scores.get(item_key) for scores in rater_scores_list if item_key in scores]

                    if gt_values:
                        gt_median = statistics.median(gt_values)
                        sorted_vals = sorted(gt_values)
                        q1 = sorted_vals[len(gt_values)//4]
                        q3 = sorted_vals[3*len(gt_values)//4]

                        in_range_flag = q1 <= lucas_pred[item_key] <= q3

                        rows.append({
                            "Framework": "LUCAS",
                            "Item": item_key,
                            "Predicted": lucas_pred[item_key],
                            "Median": f"{gt_median:.1f}",
                            "Q1": f"{q1:.1f}",
                            "Q3": f"{q3:.1f}",
                            "In_Range": "✓" if in_range_flag else "✗",
                        })

    # Clinical results - GSLP
    gslp_pred = {k: v for k, v in clinical_pred.items() if k.startswith("LP_GS_")}
    if gslp_pred:
        gt_data = load_gt_for_video("GSLP.csv", gt_video)
        if gt_data:
            result = compute_alignment(gslp_pred, gt_data)
            if result:
                for d in result["details"]:
                    rows.append({
                        "Framework": "GSLP",
                        "Item": d["item"],
                        "Predicted": d["predicted"],
                        "Median": f"{d['median']:.1f}",
                        "Q1": f"{d['q1']:.1f}",
                        "Q3": f"{d['q3']:.1f}",
                        "In_Range": "✓" if d["in_range"] else "✗",
                    })

    # Clinical results - LP_Aufklaerung
    lp_auf_pred = {k: v for k, v in clinical_pred.items() if k.startswith("LP_") and not k.startswith("LP_GS_")}
    if lp_auf_pred:
        gt_data = load_gt_for_video("LP_Aufklaerung.csv", gt_video)
        if gt_data:
            result = compute_alignment(lp_auf_pred, gt_data)
            if result:
                for d in result["details"]:
                    rows.append({
                        "Framework": "LP_Aufklaerung",
                        "Item": d["item"],
                        "Predicted": d["predicted"],
                        "Median": f"{d['median']:.1f}",
                        "Q1": f"{d['q1']:.1f}",
                        "Q3": f"{d['q3']:.1f}",
                        "In_Range": "✓" if d["in_range"] else "✗",
                    })

    # SPIKES results
    if spikes_pred:
        gt_data = load_gt_for_video("spikes.csv", gt_video)
        if gt_data:
            # Manually compute alignment for SPIKES using SPIKES_MAPPING
            mappable_items = {item_id: pred for item_id, pred in spikes_pred.items()
                            if item_id in SPIKES_MAPPING}

            if mappable_items:
                in_range = 0
                total = len(mappable_items)

                for item_id, pred_score in mappable_items.items():
                    gt_col = SPIKES_MAPPING[item_id]

                    # Find corresponding GT values for this item
                    gt_values = []
                    for gt_row in gt_data:
                        if gt_col in gt_row:
                            try:
                                gt_values.append(int(gt_row[gt_col]))
                            except (ValueError, TypeError):
                                pass

                    if not gt_values:
                        continue

                    # Compute consensus
                    gt_median = statistics.median(gt_values)
                    sorted_vals = sorted(gt_values)
                    q1 = sorted_vals[len(gt_values)//4]
                    q3 = sorted_vals[3*len(gt_values)//4]

                    in_range_flag = q1 <= pred_score <= q3
                    if in_range_flag:
                        in_range += 1

                    rows.append({
                        "Framework": "SPIKES",
                        "Item": item_id,
                        "Predicted": pred_score,
                        "Median": f"{gt_median:.1f}",
                        "Q1": f"{q1:.1f}",
                        "Q3": f"{q3:.1f}",
                        "In_Range": "✓" if in_range_flag else "✗",
                    })

    # Diabetes clinical results
    diabetes_pred = {k: v for k, v in clinical_pred.items() if k.startswith("DM_")}
    if diabetes_pred:
        gt_data = load_gt_for_video("Diabetes.csv", gt_video)
        if gt_data:
            # Manually compute alignment for Diabetes using DIABETES_MAPPING
            mappable_items = {item_id: pred for item_id, pred in diabetes_pred.items()
                            if item_id in DIABETES_MAPPING}

            if mappable_items:
                for item_id, pred_score in mappable_items.items():
                    gt_col = DIABETES_MAPPING[item_id]

                    # Find corresponding GT values for this item
                    gt_values = []
                    for gt_row in gt_data:
                        if gt_col in gt_row:
                            try:
                                gt_values.append(int(gt_row[gt_col]))
                            except (ValueError, TypeError):
                                pass

                    if not gt_values:
                        continue

                    # Compute consensus
                    gt_median = statistics.median(gt_values)
                    sorted_vals = sorted(gt_values)
                    q1 = sorted_vals[len(gt_values)//4]
                    q3 = sorted_vals[3*len(gt_values)//4]

                    in_range_flag = q1 <= pred_score <= q3

                    rows.append({
                        "Framework": "Diabetes",
                        "Item": item_id,
                        "Predicted": pred_score,
                        "Median": f"{gt_median:.1f}",
                        "Q1": f"{q1:.1f}",
                        "Q3": f"{q3:.1f}",
                        "In_Range": "✓" if in_range_flag else "✗",
                    })

    # Write CSV
    if rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Framework", "Item", "Predicted", "Median", "Q1", "Q3", "In_Range"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Results saved to {output_file} ({len(rows)} items)")
    else:
        print("No results to save")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compare pipeline predictions with ground truth")
    parser.add_argument("session_name", help="Session name (e.g., session_005)")
    parser.add_argument("--output", "-o", help="Save results to CSV file (default: results/results_{session}.csv)")

    args = parser.parse_args()

    # Default output path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    default_output = results_dir / f"results_{args.session_name}.csv"

    output_file = args.output if args.output else str(default_output)
    save_results_csv(args.session_name, output_file)

    # Also show comparison in terminal
    print()
    main(args.session_name)
