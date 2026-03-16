#!/bin/bash
#
# Complete analysis workflow: Pipeline → Analysis → Dashboard
#
# Steps:
#   1. Run full pipeline on all sessions (extract LUCAS/SPIKES/Clinical)
#   2. Run unified comparison (against gold truth)
#   3. Generate interactive dashboard
#

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         COMPLETE ANALYSIS WORKFLOW                            ║"
echo "║  Pipeline → Unified Comparison → Dashboard                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# ────────────────────────────────────────────────────────────────────
# Step 1: Run Pipeline on All Sessions
# ────────────────────────────────────────────────────────────────────

echo ""
echo "STEP 1: Running pipeline on all sessions..."
echo "────────────────────────────────────────────────────────────────"

SESSIONS=(
    "session_001"
    "session_003"
    "session_005"
    "session_006"
    "session_007"
    "session_008"
)

for session in "${SESSIONS[@]}"; do
    session_path="/data/malhaskir/simdata/data/raw/$session"
    if [ -d "$session_path" ]; then
        echo "  ▶ $session"
        python pipeline.py --input "$session_path" --force >/dev/null 2>&1 || echo "    ⚠ Failed (may already be processed)"
    fi
done

echo "✓ Pipeline complete"

# ────────────────────────────────────────────────────────────────────
# Step 2: Run Unified Comparison
# ────────────────────────────────────────────────────────────────────

echo ""
echo "STEP 2: Running unified comparison..."
echo "────────────────────────────────────────────────────────────────"

python analysis_unified_comparison.py

echo ""

# ────────────────────────────────────────────────────────────────────
# Step 3: Show Results Summary
# ────────────────────────────────────────────────────────────────────

echo "STEP 3: Results Summary"
echo "────────────────────────────────────────────────────────────────"

if [ -f "unified_comparison_results.csv" ]; then
    echo ""
    echo "Unified Comparison Results:"
    column -t -s',' unified_comparison_results.csv | head -10
    echo ""
fi

echo "✓ Analysis workflow complete!"
echo ""
echo "Output files:"
echo "  • unified_comparison_results.csv (detailed comparison)"
echo ""
echo "Next steps:"
echo "  • Review results: cat unified_comparison_results.csv"
echo "  • Test strictness levels: python analysis_strictness_comparison.py"
echo "  • Generate dashboard: python analysis_dashboard.py"
