#!/usr/bin/env python3
"""
Create publication-ready results comparison figure (Nature journal format).

Generates a multi-panel figure showing:
- LUCAS alignment (Panel A)
- GSLP alignment (Panel B)
- LP_Aufklaerung alignment (Panel C)
- Summary statistics (Panel D)
- Deviation patterns (Panel E)

Usage:
    python create_results_figure.py session_005

Output:
    figures/results_session_005.pdf (300 DPI, Nature format)
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys
import argparse
import statistics


# Nature journal guidelines
NATURE_WIDTH_MM = 180  # Single column
DPI = 300
FIGSIZE = (NATURE_WIDTH_MM/25.4, 10)  # Convert mm to inches
FONT_SIZES = {
    "title": 10,
    "label": 10,
    "tick": 8,
    "legend": 8,
}
COLORS = {
    "in_range": "#2ecc71",  # Green
    "out_range": "#e74c3c",  # Red
    "median": "#3498db",    # Blue
    "q1q3": "#95a5a6",      # Gray
}


def load_results_csv(session_name):
    """Load results CSV file"""
    results_path = Path("results") / f"results_{session_name}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    data = []
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "Framework": row["Framework"],
                "Item": row["Item"],
                "Predicted": float(row["Predicted"]),
                "Median": float(row["Median"]),
                "Q1": float(row["Q1"]),
                "Q3": float(row["Q3"]),
                "In_Range": row["In_Range"],
            })
    return data


def parse_results(data):
    """Parse results by framework"""
    lucas = [d for d in data if d["Framework"] == "LUCAS"]
    gslp = [d for d in data if d["Framework"] == "GSLP"]
    lp_auf = [d for d in data if d["Framework"] == "LP_Aufklaerung"]

    return lucas, gslp, lp_auf


def calculate_statistics(data):
    """Calculate alignment statistics"""
    in_range = sum(1 for d in data if d["In_Range"] == "✓")
    total = len(data)
    alignment_pct = (in_range / total * 100) if total > 0 else 0

    return {
        "in_range": in_range,
        "total": total,
        "alignment_pct": alignment_pct,
    }


def plot_framework_panel(ax, data, title, label):
    """Plot alignment for a framework (LUCAS, GSLP, or LP_Aufklaerung)"""
    # Sort by item
    data = sorted(data, key=lambda x: x["Item"])

    n_items = len(data)
    x = np.arange(n_items)

    items = [d["Item"] for d in data]
    predicted = [d["Predicted"] for d in data]
    median = [d["Median"] for d in data]
    q1 = [d["Q1"] for d in data]
    q3 = [d["Q3"] for d in data]
    in_range = [d["In_Range"] == "✓" for d in data]

    # Plot IQR (Q1-Q3) as background bands
    for i, (q1_val, q3_val, is_in_range) in enumerate(zip(q1, q3, in_range)):
        color = COLORS["in_range"] if is_in_range else COLORS["out_range"]
        alpha = 0.3
        ax.barh(i, q3_val - q1_val, left=q1_val, height=0.6,
                color=color, alpha=alpha, zorder=1)

    # Plot median line
    ax.scatter(median, x, s=100, marker="|", color=COLORS["median"],
               zorder=3, linewidths=2, label="Median")

    # Plot predictions
    colors = [COLORS["in_range"] if is_in else COLORS["out_range"]
              for is_in in in_range]
    ax.scatter(predicted, x, s=80, marker="o", c=colors, zorder=4,
               edgecolors="black", linewidths=1)

    # Formatting
    ax.set_yticks(x)
    ax.set_yticklabels(items, fontsize=FONT_SIZES["tick"])
    ax.set_xlabel("Score", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. {title}", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add alignment percentage
    stats = calculate_statistics(data)
    ax.text(0.98, 0.05, f"Alignment: {stats['alignment_pct']:.1f}%\n({stats['in_range']}/{stats['total']})",
            transform=ax.transAxes, fontsize=FONT_SIZES["legend"],
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_summary_statistics(ax, lucas, gslp, lp_auf, label):
    """Plot summary statistics panel"""
    frameworks = ["LUCAS", "GSLP", "LP_Aufklaerung"]
    stats_list = [calculate_statistics(lucas),
                  calculate_statistics(gslp),
                  calculate_statistics(lp_auf)]

    alignments = [s["alignment_pct"] for s in stats_list]
    in_ranges = [s["in_range"] for s in stats_list]
    totals = [s["total"] for s in stats_list]

    # Bar plot
    x = np.arange(len(frameworks))
    width = 0.35

    colors_bar = [COLORS["in_range"], COLORS["in_range"], COLORS["out_range"]]
    bars = ax.bar(x, alignments, width, label="Alignment %", color=colors_bar)

    # Add value labels on bars
    for i, (bar, align) in enumerate(zip(bars, alignments)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f"{align:.1f}%\n({in_ranges[i]}/{totals[i]})",
                ha="center", va="bottom", fontsize=FONT_SIZES["tick"])

    ax.set_ylabel("Alignment (%)", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Summary Statistics", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=FONT_SIZES["tick"], rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=50, color="red", linestyle=":", alpha=0.5, label="50% threshold")
    ax.legend(fontsize=FONT_SIZES["legend"], loc="upper right")


def plot_deviations(ax, lucas, gslp, lp_auf, label):
    """Plot deviation analysis"""
    all_data = lucas + gslp + lp_auf

    # Calculate absolute deviation from median
    deviations_in = []
    deviations_out = []

    for d in all_data:
        deviation = abs(d["Predicted"] - d["Median"])
        if d["In_Range"] == "✓":
            deviations_in.append(deviation)
        else:
            deviations_out.append(deviation)

    in_range_count = len(deviations_in)
    out_range_count = len(deviations_out)

    # Violin plot
    data_to_plot = [deviations_in, deviations_out]
    parts = ax.violinplot(data_to_plot, positions=[0, 1],
                          showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        if i == 0:
            pc.set_facecolor(COLORS["in_range"])
        else:
            pc.set_facecolor(COLORS["out_range"])
        pc.set_alpha(0.6)

    ax.set_ylabel("Absolute Deviation", fontsize=FONT_SIZES["label"])
    ax.set_title(f"{label}. Deviation Distribution", fontsize=FONT_SIZES["title"],
                 fontweight="bold", loc="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"In Range\n(n={in_range_count})",
                        f"Out of Range\n(n={out_range_count})"],
                       fontsize=FONT_SIZES["tick"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def create_figure(session_name):
    """Create complete multi-panel figure"""
    # Load data
    data = load_results_csv(session_name)
    lucas, gslp, lp_auf = parse_results(data)

    # Create figure with Nature guidelines
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    # Create grid: 5 panels
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # Plot panels
    plot_framework_panel(ax_a, lucas, "LUCAS Assessment", "A")
    plot_framework_panel(ax_b, gslp, "GSLP Assessment", "B")
    plot_framework_panel(ax_c, lp_auf, "LP_Aufklaerung Assessment", "C")
    plot_summary_statistics(ax_d, lucas, gslp, lp_auf, "D")
    plot_deviations(ax_e, lucas, gslp, lp_auf, "E")

    # Overall title
    fig.suptitle(f"Pipeline Predictions vs Ground Truth Comparison ({session_name})",
                 fontsize=FONT_SIZES["title"] + 2, fontweight="bold", y=0.98)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["in_range"], alpha=0.6,
                       label="In consensus range (IQR)"),
        mpatches.Patch(facecolor=COLORS["out_range"], alpha=0.6,
                       label="Outside consensus range"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, label="Prediction", markeredgecolor="black", markeredgewidth=1),
        plt.Line2D([0], [0], marker="|", color=COLORS["median"], markersize=12,
                   linewidth=2, label="Median rater score"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=FONT_SIZES["legend"], bbox_to_anchor=(0.5, -0.02), frameon=True)

    return fig


def main(session_name):
    """Main execution"""
    print(f"Creating results figure for {session_name}...")

    # Create figure
    fig = create_figure(session_name)

    # Save figure
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"results_{session_name}.pdf"

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", format="pdf")
    print(f"✅ Figure saved to: {output_path}")

    # Also save as PNG for preview
    png_path = output_dir / f"results_{session_name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
    print(f"✅ Preview saved to: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create publication-ready results figure")
    parser.add_argument("session_name", help="Session name (e.g., session_005)")

    args = parser.parse_args()
    main(args.session_name)
