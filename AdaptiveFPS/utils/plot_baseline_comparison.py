"""
plot_baseline_comparison.py
─────────────────────────────
Compares FixedFPS, rule-based (GT-Curvature / Inverse GT-Curvature), Random,
and selected AdaptiveFPS policies in clean scientific bar plots, reading
directly from a self-contained statistics_summary.csv.

Example
-------
python plot_baseline_comparison.py \
    --csv AdaptiveFPS/eval/f1_aut/statistics_summary.csv \
    --adaptive-policy adaptive_fc_0.075_bud_300.0

--adaptive-policy is repeatable; with none given, only the fixed/rule-based/
random baselines are plotted. Figures are saved in the same directory as
the input CSV.
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

# Same restrained, print-safe palette family used elsewhere in this repo's
# sensing-trade-off figures, extended to one color per policy *category*
# here (rather than per budget, since that's the dimension this figure
# compares across).
CATEGORY_COLORS = {
    0: "#08519c",  # Fixed - blue
    1: "#969696",  # Random - neutral gray
    2: "#fd8d3c",  # Inverse GT-Curvature - light orange
    3: "#e6550d",  # GT-Curvature - dark orange
    4: "#31a354",  # Adaptive (selected) - green
}

FIXED_RE = re.compile(r"^fixed_(\d+)Hz$")
RANDOM_RE = re.compile(r"^random")
ADAPTIVE_RE = re.compile(r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)")
SEED_RE = re.compile(r"seed_(\d+)")


def format_fc_compact(frame_cost):
    """Compact frame-cost label for the small Adaptive tick label, e.g.
    0.075 -> '.075', 0.0 -> '0' (matches the leading-zero-dropped style
    requested for these space-constrained multiline tick labels)."""
    if frame_cost == 0.0:
        return "0"
    text = f"{frame_cost:g}"
    return text[1:] if text.startswith("0.") else text


def classify_policy(policy, adaptive_policies):
    """Return (category_rank, sort_subkey, multiline_label) for a policy we
    want plotted, or None to exclude it. category_rank fixes the logical
    ordering: Fixed -> Random -> Inverse GT-Curvature -> GT-Curvature ->
    Adaptive (only the explicitly --adaptive-policy-selected ones)."""
    match = FIXED_RE.match(policy)
    if match:
        hz = int(match.group(1))
        return (0, hz, f"Fixed\n{hz} Hz")

    if RANDOM_RE.match(policy):
        seed_match = SEED_RE.search(policy)
        seed = int(seed_match.group(1)) if seed_match else 0
        return (1, seed, "Random\n5/10 Hz")

    if policy == "gt_curvature_inverse":
        return (2, 0, "Inverse GT-\nCurvature")

    if policy == "gt_curvature":
        return (3, 0, "GT-\nCurvature")

    if policy in adaptive_policies:
        match = ADAPTIVE_RE.search(policy)
        if match is None:
            print(f"Warning: --adaptive-policy '{policy}' does not match the expected "
                  f"adaptive_fc_<fc>_bud_<budget> naming, skipping")
            return None
        frame_cost, budget = float(match.group(1)), float(match.group(2))
        order = adaptive_policies.index(policy)
        return (4, order, f"Adaptive\nfc={format_fc_compact(frame_cost)}, B={int(budget)}")

    return None


def build_selection(df, adaptive_policies):
    """Classify/label/order the policies to plot; unselected adaptive rows
    and anything unrecognized are silently excluded (not an error - this
    figure is deliberately a curated subset, not every row in the CSV)."""
    classified = df["policy"].apply(lambda p: classify_policy(p, adaptive_policies))

    df = df.copy()
    df["rank"] = [c[0] if c else np.nan for c in classified]
    df["subkey"] = [c[1] if c else np.nan for c in classified]
    df["label"] = [c[2] if c else None for c in classified]
    df = df.dropna(subset=["rank"]).copy()
    df["rank"] = df["rank"].astype(int)

    # Disambiguate multiple random-seed rows only if more than one is present.
    random_rows = df["rank"] == 1
    if random_rows.sum() > 1:
        df.loc[random_rows, "label"] = [
            f"{label} (s{seed})" for label, seed in zip(df.loc[random_rows, "label"], df.loc[random_rows, "subkey"])
        ]

    found_adaptive = set(df.loc[df["rank"] == 4, "policy"])
    for policy in adaptive_policies:
        if policy not in found_adaptive:
            print(f"Warning: --adaptive-policy '{policy}' not found in the CSV, skipping")

    return df.sort_values(["rank", "subkey"], kind="stable").reset_index(drop=True)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_metric(df, value_col, err_col, ylabel, title, filename_stem, out_dir, ylim=None, y_tick_step=None):
    x = np.arange(len(df))
    colors = [CATEGORY_COLORS[rank] for rank in df["rank"]]
    errs = df[err_col] if err_col and err_col in df.columns else None

    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(df) + 2.0), 5.2))

    ax.bar(
        x,
        df[value_col],
        width=0.6,
        yerr=errs,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0} if errs is not None else {},
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_xlabel("Sensing Policy")

    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))

    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()

    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare FixedFPS/rule-based/random/selected-AdaptiveFPS policies in bar plots."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to statistics_summary.csv")
    parser.add_argument(
        "--adaptive-policy",
        action="append",
        default=[],
        metavar="POLICY_NAME",
        help="Adaptive policy name from the 'policy' column to include (repeatable), "
             "e.g. --adaptive-policy adaptive_fc_0.075_bud_300.0",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = csv_path.parent
    raw_df = pd.read_csv(csv_path)

    if "policy" not in raw_df.columns:
        raise ValueError(f"'{csv_path}' has no 'policy' column.")

    df = build_selection(raw_df, args.adaptive_policy)
    if df.empty:
        raise ValueError(f"No matching policies found to plot in {csv_path}.")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    saved_paths = []

    obs_png, obs_pdf = plot_metric(
        df, "mean_observation_reduction_percent", "std_observation_reduction_percent",
        "Observation Skipping Rate (%)", "Observation Skipping Rate by Sensing Policy",
        "baseline_observation_skipping", out_dir, ylim=(0, 100), y_tick_step=20,
    )
    saved_paths += [obs_png, obs_pdf]

    succ_png, succ_pdf = plot_metric(
        df, "success_rate_percent", None,
        "Success Rate (%)", "Success Rate by Sensing Policy",
        "baseline_success_rate", out_dir, ylim=(0, 105), y_tick_step=20,
    )
    saved_paths += [succ_png, succ_pdf]

    if "mean_fps" in raw_df.columns:
        err_col = "std_mean_fps" if "std_mean_fps" in raw_df.columns else None
        fps_max = df["mean_fps"].max()
        fps_png, fps_pdf = plot_metric(
            df, "mean_fps", err_col,
            "Mean Sensing Rate (Hz)", "Mean Sensing Rate by Sensing Policy",
            "baseline_mean_sensing_rate", out_dir, ylim=(0, fps_max * 1.15),
        )
        saved_paths += [fps_png, fps_pdf]
    else:
        print(
            "Note: 'mean_fps' is not a column in statistics_summary.csv - "
            "compute_statistics.py does not currently compute a mean-FPS metric, "
            "so the Mean Sensing Rate figure is being skipped rather than deriving "
            "an approximate value from the observation-skipping percentage."
        )

    print(f"Policies plotted: {len(df)}")
    print()
    print("Saved figures:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
