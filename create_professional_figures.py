#!/usr/bin/env python3
"""
Create professional publication-ready figures for results analysis.

Generates:
1. Per-session detailed comparison (5-panel figure)
2. Cross-session summary analysis (multi-framework comparison)

Uses Nature journal guidelines:
- 180mm single column width
- 300 DPI for publication
- Professional color schemes
- Multiple plot types: scatter, box, violin, heatmap, line

Usage:
    python create_professional_figures.py session_005              # Single session
    python create_professional_figures.py --all-sessions           # All sessions
    python create_professional_figures.py --cross-session          # Cross-session summary

Output:
    figures/results_session_005_detailed.pdf
    figures/cross_session_summary.pdf
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import sys
import argparse
import glob


# Nature journal guidelines
NATURE_WIDTH_MM = 180
DPI = 300
FIGSIZE_SINGLE = (NATURE_WIDTH_MM/25.4, 10)
FIGSIZE_WIDE = (NATURE_WIDTH_MM/25.4, 12)
FONT_SIZES = {
    "title": 11,
    "label": 10,
    "tick": 8,
    "legend": 8,
    "annotation": 7,
}
COLORS = {
    # Nature journal palette - professional, muted colors
    "in_range": "#2A9D8F",      # Teal (in consensus)
    "out_range": "#E76F51",      # Burnt orange (out of consensus)
    "median": "#264653",         # Dark blue (median line)
    "q1q3": "#E9C46A",           # Gold (IQR bands)
    "lucas": "#264653",          # Dark navy blue
    "gslp": "#2A9D8F",           # Teal
    "lp_auf": "#E76F51",         # Burnt orange
    "neutral": "#999999",        # Gray
}


def load_results_csv(session_name):
    """Load results CSV file"""
    results_path = Path("results") / f"results_{session_name}.csv"
    if not results_path.exists():
        return None

    data = []
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "Session": session_name,
                "Framework": row["Framework"],
                "Item": row["Item"],
                "Predicted": float(row["Predicted"]),
                "Median": float(row["Median"]),
                "Q1": float(row["Q1"]),
                "Q3": float(row["Q3"]),
                "In_Range": row["In_Range"] == "✓",
            })
    return data


def get_all_sessions():
    """Find all session results"""
    csv_files = glob.glob("results/results_session_*.csv")
    sessions = [Path(f).stem.replace("results_", "") for f in csv_files]
    return sorted(sessions)


def calculate_stats(data):
    """Calculate alignment statistics"""
    if not data:
        return None
    in_range = sum(1 for d in data if d["In_Range"])
    total = len(data)
    return {
        "in_range": in_range,
        "total": total,
        "alignment": (in_range / total * 100) if total > 0 else 0,
    }


def plot_scatter_predicted_vs_raters(ax, data, title, label):
    """Scatter plot: Predicted vs Median rater scores"""
    predicted = [d["Predicted"] for d in data]
    median = [d["Median"] for d in data]
    colors = [COLORS["in_range"] if d["In_Range"] else COLORS["out_range"] for d in data]

    ax.scatter(median, predicted, c=colors, s=100, alpha=0.6, edgecolors="black", linewidth=1)

    # Perfect agreement line
    ax.plot([0, 2], [0, 2], "k--", alpha=0.3, linewidth=1)

    # Add ±0.5 boundaries
    ax.fill_between([0, 2], [-0.5, 1.5], [0.5, 2.5], alpha=0.1, color=COLORS["neutral"])

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    ax.set_xlabel("Median Rater Score", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Pipeline Prediction", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. {title}", fontsize=FONT_SIZES["title"], fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    # Add stats
    stats = calculate_stats(data)
    ax.text(0.05, 0.95, f"{stats['alignment']:.1f}% aligned\nn={stats['total']}",
            transform=ax.transAxes, fontsize=FONT_SIZES["legend"],
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.8, edgecolor=COLORS["neutral"], linewidth=1))


def plot_box_plots_by_framework(ax, all_data, title, label):
    """Box plot: Distribution by framework"""
    frameworks = ["LUCAS", "GSLP", "LP_Aufklaerung"]
    data_by_fw = {fw: [d for d in all_data if d["Framework"] == fw] for fw in frameworks}

    deviations = []
    labels = []
    for fw in frameworks:
        if data_by_fw[fw]:
            devs = [abs(d["Predicted"] - d["Median"]) for d in data_by_fw[fw]]
            deviations.append(devs)
            labels.append(fw)

    bp = ax.boxplot(deviations, labels=labels, patch_artist=True,
                    medianprops=dict(color=COLORS["median"], linewidth=2),
                    boxprops=dict(facecolor=COLORS["in_range"], alpha=0.6),
                    whiskerprops=dict(color=COLORS["neutral"]),
                    capprops=dict(color=COLORS["neutral"]))

    ax.set_ylabel("Absolute Deviation", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Deviation Distribution by Framework",
                 fontsize=FONT_SIZES["title"], fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(labels, fontsize=FONT_SIZES["tick"], rotation=15, ha="right")


def plot_item_level_heatmap(ax, data, title, label):
    """Heatmap: Alignment by item"""
    # Group by item
    items = sorted(set(d["Item"] for d in data))
    frameworks = sorted(set(d["Framework"] for d in data))

    # Create matrix
    matrix = np.zeros((len(frameworks), len(items)))
    for i, fw in enumerate(frameworks):
        for j, item in enumerate(items):
            item_data = [d for d in data if d["Framework"] == fw and d["Item"] == item]
            if item_data:
                d = item_data[0]
                # Score: 1 if in range, 0 if out
                matrix[i, j] = 1 if d["In_Range"] else 0

    # Plot heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(items)))
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_xticklabels(items, fontsize=FONT_SIZES["tick"], rotation=45, ha="right")
    ax.set_yticklabels(frameworks, fontsize=FONT_SIZES["tick"])

    ax.set_title(f"{label}. Item-Level Alignment", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")

    # Add text annotations
    for i in range(len(frameworks)):
        for j in range(len(items)):
            text = "✓" if matrix[i, j] == 1 else "✗"
            ax.text(j, i, text, ha="center", va="center", color="black",
                   fontsize=FONT_SIZES["annotation"], fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("In Range", fontsize=FONT_SIZES["label"])


def plot_deviation_analysis(ax, data, title, label):
    """Violin plot: Deviation distribution"""
    frameworks = ["LUCAS", "GSLP", "LP_Aufklaerung"]
    data_by_fw = {fw: [d for d in data if d["Framework"] == fw] for fw in frameworks}

    deviations = []
    labels = []
    for fw in frameworks:
        if data_by_fw[fw]:
            devs = [abs(d["Predicted"] - d["Median"]) for d in data_by_fw[fw]]
            deviations.append(devs)
            labels.append(fw)

    # Violin plot
    parts = ax.violinplot(deviations, positions=range(len(deviations)),
                         showmeans=True, showmedians=True)

    # Color violins
    color_map = {"LUCAS": COLORS["lucas"], "GSLP": COLORS["gslp"],
                 "LP_Aufklaerung": COLORS["lp_auf"]}
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(color_map[labels[i]])
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZES["tick"], rotation=15, ha="right")
    ax.set_ylabel("Absolute Deviation", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Deviation Distribution (Violin)", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3)


def create_per_session_figure(session_name):
    """Create detailed per-session 5-panel figure"""
    data = load_results_csv(session_name)
    if not data:
        print(f"❌ No data found for {session_name}")
        return None

    # Parse by framework
    lucas = [d for d in data if d["Framework"] == "LUCAS"]
    gslp = [d for d in data if d["Framework"] == "GSLP"]
    lp_auf = [d for d in data if d["Framework"] == "LP_Aufklaerung"]

    # Create figure
    fig = plt.figure(figsize=FIGSIZE_SINGLE, dpi=DPI)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Plot panels
    plot_scatter_predicted_vs_raters(ax1, lucas, "LUCAS", "A")
    plot_scatter_predicted_vs_raters(ax2, gslp, "GSLP", "B")
    plot_scatter_predicted_vs_raters(ax3, lp_auf, "LP_Aufklaerung", "C")
    plot_box_plots_by_framework(ax4, data, "Deviation Analysis", "D")
    plot_item_level_heatmap(ax5, data, "Alignment Matrix", "E")

    fig.suptitle(f"Detailed Results Analysis: {session_name}",
                 fontsize=FONT_SIZES["title"] + 2, fontweight="bold", y=0.98)

    return fig


def create_cross_session_figure():
    """Create cross-session summary figure"""
    sessions = get_all_sessions()
    if not sessions:
        print("❌ No session data found")
        return None

    # Load all data
    all_data = []
    for session in sessions:
        data = load_results_csv(session)
        if data:
            all_data.extend(data)

    if not all_data:
        print("❌ No data loaded")
        return None

    # Create figure
    fig = plt.figure(figsize=FIGSIZE_WIDE, dpi=DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Plot 1: Alignment by session and framework
    plot_cross_session_alignment(ax1, all_data, sessions, "A")

    # Plot 2: Framework comparison
    plot_framework_comparison(ax2, all_data, "B")

    # Plot 3: Item consistency across sessions
    plot_item_consistency(ax3, all_data, sessions, "C")

    fig.suptitle("Cross-Session Summary: Pipeline Performance",
                 fontsize=FONT_SIZES["title"] + 2, fontweight="bold", y=0.98)

    return fig


def plot_cross_session_alignment(ax, data, sessions, label):
    """Line plot: Alignment trend across sessions"""
    frameworks = ["LUCAS", "GSLP", "LP_Aufklaerung"]

    for fw in frameworks:
        alignments = []
        for session in sessions:
            fw_data = [d for d in data if d["Session"] == session and d["Framework"] == fw]
            if fw_data:
                stats = calculate_stats(fw_data)
                alignments.append(stats["alignment"])
            else:
                alignments.append(None)

        # Plot line
        valid_indices = [i for i, x in enumerate(alignments) if x is not None]
        valid_sessions = [sessions[i] for i in valid_indices]
        valid_alignments = [alignments[i] for i in valid_indices]

        color = COLORS.get(fw.lower().replace("_", "_"), "#3498db")
        ax.plot(valid_sessions, valid_alignments, marker="o", linewidth=2,
                label=fw, color=color, markersize=8)

    ax.set_xlabel("Session", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Alignment (%)", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Alignment Trend Across Sessions", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color=COLORS["neutral"], linestyle=":", alpha=0.5, label="50% threshold")
    ax.legend(fontsize=FONT_SIZES["legend"], loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)


def plot_framework_comparison(ax, data, label):
    """Violin plot: Framework comparison across all sessions"""
    frameworks = ["LUCAS", "GSLP", "LP_Aufklaerung"]
    deviations = []
    labels = []

    for fw in frameworks:
        fw_data = [d for d in data if d["Framework"] == fw]
        if fw_data:
            devs = [abs(d["Predicted"] - d["Median"]) for d in fw_data]
            deviations.append(devs)
            labels.append(fw)

    parts = ax.violinplot(deviations, positions=range(len(deviations)),
                         showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        color_map = {"LUCAS": COLORS["lucas"], "GSLP": COLORS["gslp"],
                     "LP_Aufklaerung": COLORS["lp_auf"]}
        pc.set_facecolor(color_map[labels[i]])
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZES["tick"], rotation=15, ha="right")
    ax.set_ylabel("Absolute Deviation", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Framework Comparison", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3)


def plot_item_consistency(ax, data, sessions, label):
    """Heatmap: Item consistency across sessions"""
    # Get unique items
    all_items = sorted(set(d["Item"] for d in data))[:15]  # Top 15 items for readability

    # Create matrix: sessions x items
    matrix = np.zeros((len(sessions), len(all_items)))
    for i, session in enumerate(sessions):
        for j, item in enumerate(all_items):
            item_data = [d for d in data if d["Session"] == session and d["Item"] == item]
            if item_data:
                # Calculate alignment for this item
                in_range = sum(1 for d in item_data if d["In_Range"])
                matrix[i, j] = in_range / len(item_data) * 100 if item_data else 0

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(all_items)))
    ax.set_yticks(np.arange(len(sessions)))
    ax.set_xticklabels(all_items, fontsize=FONT_SIZES["tick"], rotation=45, ha="right")
    ax.set_yticklabels(sessions, fontsize=FONT_SIZES["tick"])

    ax.set_title(f"{label}. Item Consistency (Alignment %)", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Alignment %")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Create professional results figures")
    parser.add_argument("session", nargs="?", default=None, help="Session name (e.g., session_005)")
    parser.add_argument("--all-sessions", action="store_true", help="Create figures for all sessions")
    parser.add_argument("--cross-session", action="store_true", help="Create cross-session summary")

    args = parser.parse_args()

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Create per-session figures
    if args.session:
        print(f"Creating per-session figure for {args.session}...")
        fig = create_per_session_figure(args.session)
        if fig:
            output_path = output_dir / f"results_{args.session}_detailed.pdf"
            fig.savefig(output_path, dpi=DPI, bbox_inches="tight", format="pdf")
            print(f"✅ Figure saved: {output_path}")

            png_path = output_dir / f"results_{args.session}_detailed.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
            print(f"✅ Preview saved: {png_path}")
            plt.close(fig)

    # Create figures for all sessions
    if args.all_sessions:
        sessions = get_all_sessions()
        for session in sessions:
            print(f"Creating figure for {session}...")
            fig = create_per_session_figure(session)
            if fig:
                output_path = output_dir / f"results_{session}_detailed.pdf"
                fig.savefig(output_path, dpi=DPI, bbox_inches="tight", format="pdf")
                png_path = output_dir / f"results_{session}_detailed.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
                plt.close(fig)
        print(f"✅ Created figures for {len(sessions)} sessions")

    # Create cross-session summary
    if args.cross_session or (not args.session and not args.all_sessions):
        print("Creating cross-session summary...")
        fig = create_cross_session_figure()
        if fig:
            output_path = output_dir / "cross_session_summary.pdf"
            fig.savefig(output_path, dpi=DPI, bbox_inches="tight", format="pdf")
            print(f"✅ Figure saved: {output_path}")

            png_path = output_dir / "cross_session_summary.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
            print(f"✅ Preview saved: {png_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
