"""
plot_sensing_trade_off_bars.py
────────────────────────────────
Decomposes the self-contained statistics_summary.csv into two independent,
publication-style grouped bar plots:

    1. Observation Skipping Rate (%) vs. Frame Cost
    2. Success Rate (%) vs. Frame Cost

Example
-------
python plot_sensing_trade_off_bars.py \
    --csv AdaptiveFPS/eval/f1_aut/statistics_summary.csv

The figures are saved in the same directory as the input CSV.
"""

import argparse
import re
from pathlib import Path

import matplotlib

# Required for headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

BUDGETS = [300, 150, 100]

# Same restrained, print-safe palette family used elsewhere in this repo's
# sensing-trade-off figures, extended with a third hue for Budget = 100.
BUDGET_COLORS = {
    300: "#08519c",
    150: "#e6550d",
    100: "#31a354",
}

BAR_WIDTH = 0.23
BUDGET_OFFSETS = {300: -BAR_WIDTH, 150: 0.0, 100: BAR_WIDTH}


# BUDGETS = [500, 250, 167]

# BUDGET_COLORS = {
#     500: "#08519c",
#     250: "#e6550d",
#     167: "#31a354",
# }

# BAR_WIDTH = 0.23
# BUDGET_OFFSETS = {
#     500: -BAR_WIDTH,
#     250: 0.0,
#     167: BAR_WIDTH,
# }

# BUDGETS = [400, 200]

# BUDGET_COLORS = {
#     400: "#08519c",
#     200: "#e6550d",
# }

# BAR_WIDTH = 0.30
# BUDGET_OFFSETS = {
#     400: -BAR_WIDTH / 2,
#     200: BAR_WIDTH / 2,
# }

def parse_policy_name(policy):
    """Extract frame cost and budget from names such as
    adaptive_fc_0.075_bud_300.0 or adaptive_fc_0.075_bud_300.0_bp_10.0.
    Returns (None, None) for anything that doesn't match (e.g. fixed_*Hz)."""
    match = re.search(r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)", str(policy))
    if match is None:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def format_fc(frame_cost):
    """Concise frame-cost tick label, no unnecessary trailing zeros."""
    return "0" if frame_cost == 0.0 else f"{frame_cost:g}"


def load_statistics(csv_path):
    """Load statistics_summary.csv and keep only adaptive fc/budget
    policies, warning (not crashing) on anything that can't be parsed."""
    df = pd.read_csv(
        csv_path,
        usecols=[
            "policy",
            "mean_observation_reduction_percent",
            "std_observation_reduction_percent",
            "success_rate_percent",
        ],
    )

    parsed = df["policy"].apply(parse_policy_name)
    df["frame_cost"] = parsed.apply(lambda t: t[0])
    df["budget"] = parsed.apply(lambda t: t[1])

    unparsed = df[df["frame_cost"].isna() | df["budget"].isna()]
    for policy in unparsed["policy"]:
        print(f"Warning: could not parse '{policy}' as an adaptive fc/budget policy, skipping")

    df = df.dropna(subset=["frame_cost", "budget"]).copy()
    df["budget"] = df["budget"].round().astype(int)

    return df


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_observation_skipping(df, frame_costs, x, out_dir):
    lookup = {(row.frame_cost, row.budget): row for row in df.itertuples()}

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for budget in BUDGETS:
        xs, heights, errs = [], [], []
        for i, fc in enumerate(frame_costs):
            row = lookup.get((fc, budget))
            if row is None:
                continue
            xs.append(x[i] + BUDGET_OFFSETS[budget])
            heights.append(row.mean_observation_reduction_percent)
            errs.append(row.std_observation_reduction_percent)

        if not xs:
            continue

        ax.bar(
            xs,
            heights,
            width=BAR_WIDTH,
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            color=BUDGET_COLORS[budget],
            edgecolor="black",
            linewidth=0.7,
            label=f"Budget = {budget}",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([format_fc(fc) for fc in frame_costs])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xlabel("Frame Cost")

    ax.set_ylabel("Observation Skipping Rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))

    ax.set_title("Observation Skipping Rate vs. Frame Cost")

    style_axes(ax)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        handletextpad=0.6,
    )

    fig.tight_layout()

    png_path = out_dir / "observation_skipping_vs_frame_cost.png"
    pdf_path = out_dir / "observation_skipping_vs_frame_cost.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def plot_success_rate(df, frame_costs, x, out_dir):
    lookup = {(row.frame_cost, row.budget): row for row in df.itertuples()}

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for budget in BUDGETS:
        xs, heights = [], []
        for i, fc in enumerate(frame_costs):
            row = lookup.get((fc, budget))
            if row is None:
                continue
            xs.append(x[i] + BUDGET_OFFSETS[budget])
            heights.append(row.success_rate_percent)

        if not xs:
            continue

        # No yerr: success_rate_percent is a single-evaluation percentage,
        # not a statistic across independent seeds -- a fabricated error
        # bar would misrepresent it.
        ax.bar(
            xs,
            heights,
            width=BAR_WIDTH,
            color=BUDGET_COLORS[budget],
            edgecolor="black",
            linewidth=0.7,
            label=f"Budget = {budget}",
            zorder=3,
        )

        # A 0%-height bar is otherwise invisible (its edge collapses into
        # the x-axis spine), which would be indistinguishable from a
        # missing policy. Mark genuine zero results with a small dot at
        # the baseline instead of fabricating a positive bar height.
        zero_xs = [xi for xi, h in zip(xs, heights) if h == 0]
        if zero_xs:
            ax.scatter(
                zero_xs,
                [0] * len(zero_xs),
                marker="o",
                s=22,
                facecolor=BUDGET_COLORS[budget],
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([format_fc(fc) for fc in frame_costs])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xlabel("Frame Cost")

    ax.set_ylabel("Success Rate (%)")
    # A hair of negative headroom keeps the zero-result marker fully visible
    # instead of being clipped by the axis spine sitting exactly at y=0.
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_locator(MultipleLocator(20))

    ax.set_title("Success Rate vs. Frame Cost")

    style_axes(ax)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        handletextpad=0.6,
    )

    fig.tight_layout()

    png_path = out_dir / "success_rate_vs_frame_cost.png"
    pdf_path = out_dir / "success_rate_vs_frame_cost.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot observation-skipping and success-rate bar charts vs. frame cost."
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

    out_dir = csv_path.parent

    df = load_statistics(csv_path)

    if df.empty:
        raise ValueError(f"No adaptive fc/budget policies found in {csv_path}")

    frame_costs = sorted(df["frame_cost"].unique())
    x = np.arange(len(frame_costs))

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    obs_png, obs_pdf = plot_observation_skipping(df, frame_costs, x, out_dir)
    succ_png, succ_pdf = plot_success_rate(df, frame_costs, x, out_dir)

    budgets_included = [b for b in BUDGETS if b in set(df["budget"])]

    print(f"Adaptive policies loaded: {len(df)}")
    print(f"Frame costs plotted: {len(frame_costs)}")
    print(f"Budgets included: {', '.join(str(b) for b in budgets_included)}")
    print()
    print("Saved figures:")
    print(obs_png)
    print(obs_pdf)
    print(succ_png)
    print(succ_pdf)


if __name__ == "__main__":
    main()
