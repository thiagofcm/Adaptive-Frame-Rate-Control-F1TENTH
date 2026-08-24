"""
Plot sensing-efficiency vs. navigation-performance trade-off.

Example
-------
python plot_sensing_tradeoff.py \
    --csv AdaptiveFPS/eval/f1_aut/statistics_summary.csv

The figure is saved in the same directory as the input CSV.
"""

import argparse
import re
from pathlib import Path

import matplotlib

# Required for headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def parse_policy_name(policy):
    """
    Extract frame cost and budget from names such as:

        adaptive_fc_0.075_bud_300.0
        adaptive_fc_0.075_bud_150.0_bp_10.0
    """
    match = re.search(
        r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)",
        str(policy),
    )

    if match is None:
        return None, None

    try:
        frame_cost = float(match.group(1))
        budget = float(match.group(2))
        return frame_cost, budget
    except ValueError:
        return None, None


def find_success_column(df):
    """Find a reasonable success-rate column in summary.csv."""
    candidates = [
        "success_rate",
        "success",
        "success_percent",
        "success_percentage",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot observation reduction vs. success rate."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to statistics_summary.csv",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    root = csv_path.parent

    # ---------------------------------------------------------
    # Load sensing statistics
    # ---------------------------------------------------------

    stats = pd.read_csv(csv_path)

    if "policy" not in stats.columns:
        raise ValueError(
            "statistics_summary.csv must contain a 'policy' column."
        )

    if "mean_observation_reduction_percent" not in stats.columns:
        raise ValueError(
            "statistics_summary.csv must contain "
            "'mean_observation_reduction_percent'."
        )

    # ---------------------------------------------------------
    # Extract frame cost and budget from policy folder name
    # ---------------------------------------------------------

    parsed = stats["policy"].apply(parse_policy_name)

    stats["frame_cost"] = parsed.apply(lambda x: x[0])
    stats["budget"] = parsed.apply(lambda x: x[1])

    # Keep adaptive policies that could be parsed.
    stats = stats[
        stats["frame_cost"].notna()
        & stats["budget"].notna()
    ].copy()

    # ---------------------------------------------------------
    # Obtain success rate
    # ---------------------------------------------------------
    #
    # First check whether success rate is already present in
    # statistics_summary.csv.
    #
    # Otherwise, automatically look for summary.csv in the same
    # root directory and merge it using policy/run name.
    # ---------------------------------------------------------

    success_column = find_success_column(stats)

    if success_column is None:

        summary_path = root / "summary.csv"

        if not summary_path.exists():
            raise FileNotFoundError(
                "Success rate was not found in statistics_summary.csv "
                "and no summary.csv exists in the same directory.\n"
                f"Expected: {summary_path}"
            )

        summary = pd.read_csv(summary_path)

        success_column = find_success_column(summary)

        if success_column is None:
            raise ValueError(
                "Could not identify a success-rate column in summary.csv.\n"
                f"Available columns: {list(summary.columns)}"
            )

        # Identify the run/policy-name column.
        if "run_name" in summary.columns:
            run_column = "run_name"
        elif "policy" in summary.columns:
            run_column = "policy"
        else:
            raise ValueError(
                "Could not identify a policy/run-name column in summary.csv.\n"
                f"Available columns: {list(summary.columns)}"
            )

        success_df = summary[
            [run_column, success_column]
        ].copy()

        success_df = success_df.rename(
            columns={run_column: "policy"}
        )

        stats = stats.merge(
            success_df,
            on="policy",
            how="left",
        )

    # Normalize name after possible merge.
    if success_column != "success_rate":
        stats["success_rate"] = stats[success_column]
    else:
        stats["success_rate"] = stats["success_rate"]

    # If success is stored in [0, 1], convert to percentage.
    if stats["success_rate"].dropna().max() <= 1.0:
        stats["success_rate"] *= 100.0

    # ---------------------------------------------------------
    # Plot selection
    # ---------------------------------------------------------
    #
    # Keep B = 150 and 300, matching the reduced table.
    # Also exclude fc = 0.2 and 0.4.
    # ---------------------------------------------------------

    plot_df = stats[
        stats["budget"].isin([150.0, 300.0])
        & (stats["frame_cost"] <= 0.10)
    ].copy()

    plot_df = plot_df.sort_values(
        ["budget", "frame_cost"]
    )

    if plot_df.empty:
        raise ValueError("No policies remained after filtering.")

    # ---------------------------------------------------------
    # Publication-style figure settings
    # ---------------------------------------------------------

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    # Distinct, print-safe marker/color per budget series.
    budget_style = {
        300.0: {"marker": "o", "color": "#08519c"},
        150.0: {"marker": "s", "color": "#e6550d"},
    }

    # Per-(budget, frame_cost) text offsets (in points) for the handful of
    # policies whose default offset would collide with a nearby
    # marker/label/line in the crowded upper-right region (fc = 0.05, 0.075,
    # 0.1). Anything not listed falls back to DEFAULT_OFFSET. Only the label
    # position is adjusted -- the underlying data points are never moved.
    annotation_offsets = {
        (300, 0.02): (2, 16),
        (150, 0.0): (10, -8),
        (300, 0.05): (-8, 14),
        (150, 0.05): (-4, -14),
        (300, 0.075): (2, 16),
        (150, 0.075): (6, -16),
        (300, 0.1): (-8, -14),
        (150, 0.1): (10, 6),
    }
    default_offset = (7, 7)

    def annotation_offset(budget, frame_cost):
        return annotation_offsets.get(
            (int(round(budget)), frame_cost), default_offset
        )

    # ---------------------------------------------------------
    # Create scatter plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for budget in sorted(plot_df["budget"].unique(), reverse=True):

        subset = plot_df[
            plot_df["budget"] == budget
        ].sort_values("frame_cost")

        style = budget_style.get(budget, {"marker": "o", "color": "black"})

        # Connecting line first, kept visually secondary to the markers.
        ax.plot(
            subset["mean_observation_reduction_percent"],
            subset["success_rate"],
            linewidth=1.3,
            alpha=0.55,
            color=style["color"],
            zorder=1,
        )

        ax.scatter(
            subset["mean_observation_reduction_percent"],
            subset["success_rate"],
            marker=style["marker"],
            s=80,
            color=style["color"],
            edgecolors="black",
            linewidths=0.6,
            label=f"Budget = {int(budget)}",
            zorder=3,
        )

        # Label each point with frame cost, using a budget/frame-cost-aware
        # offset so nearby labels don't overlap each other, the markers,
        # the connecting lines, the legend, or the plot boundaries.
        for _, row in subset.iterrows():

            fc = row["frame_cost"]

            if fc == 0.0:
                fc_label = "0"
            else:
                fc_label = f"{fc:g}"

            dx, dy = annotation_offset(budget, fc)
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"

            ax.annotate(
                f"$f_c={fc_label}$",
                (
                    row["mean_observation_reduction_percent"],
                    row["success_rate"],
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha=ha,
                va=va,
                color=style["color"],
                annotation_clip=True,
                zorder=4,
            )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_xlabel("Observation Skipping Rate (%)")
    ax.set_ylabel("Success Rate (%)")

    ax.set_title("Sensing–Performance Trade-off")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
        zorder=0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Data-driven axis limits with a small margin so no marker or annotation
    # sits directly against the border. Success rate is concentrated near
    # the upper end, so the y-axis is not forced to start at 0.
    x_vals = plot_df["mean_observation_reduction_percent"]
    y_vals = plot_df["success_rate"]

    x_margin = (x_vals.max() - x_vals.min()) * 0.12
    y_margin = (y_vals.max() - y_vals.min()) * 0.25

    ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
    ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

    legend = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    png_path = root / "sensing_efficiency_tradeoff.png"
    pdf_path = root / "sensing_efficiency_tradeoff.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Policies plotted: {len(plot_df)}")
    print()
    print("Saved plots to:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()