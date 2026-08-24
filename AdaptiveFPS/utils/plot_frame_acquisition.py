"""
plot_frame_acquisition.py
──────────────────────────
Publication-style raster plot of per-control-step frame acquisition for the
first 10 evaluation laps of an AdaptiveFPS run, with an aligned success
column.

Example
-------
python plot_frame_acquisition.py \
    --eval-dir AdaptiveFPS/eval/f1_aut/adaptive_fc_0.075_bud_300.0

The figures are saved inside the evaluation directory.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

N_LAPS = 10

ACQUIRED_COLOR = "#e6550d"
NOT_ACQUIRED_COLOR = "#08519c"
NO_DATA_COLOR = "#ececec"


def parse_bool_column(series):
    """Robustly convert a boolean-like column to bool, handling True/False,
    "true"/"false" (any case), and 1/0 (numeric or string)."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def parse_policy_name(name):
    """Extract frame cost and budget from names such as
    adaptive_fc_0.075_bud_300.0 or adaptive_fc_0.075_bud_300.0_bp_10.0."""
    match = re.search(r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)", str(name))
    if match is None:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def load_laps(eval_dir):
    """Load step/frame_consumed for lap_00 .. lap_09, skipping (with a
    warning) any lap whose steps.csv is missing rather than crashing."""
    laps = []
    for lap_num in range(N_LAPS):
        steps_path = eval_dir / f"lap_{lap_num:02d}" / "steps.csv"

        if not steps_path.exists():
            print(f"Warning: {steps_path} not found, skipping lap {lap_num}")
            continue

        df = pd.read_csv(steps_path, usecols=["step", "frame_consumed"])
        df["frame_consumed"] = parse_bool_column(df["frame_consumed"])
        laps.append((lap_num, df))

    return laps


def find_success_lookup(eval_dir):
    """Locate an episode-level result file with a lap-index column and a
    success column, and return {lap_index: success_bool}.

    Success is never inferred from step data -- if nothing usable is found,
    raise an error describing exactly what was inspected.
    """
    index_candidates = ["lap_index", "lap", "episode", "episode_index"]
    success_candidates = ["success", "success_rate", "succeeded"]

    # Check episodes.csv first (the standard evaluation output), then fall
    # back to any other top-level CSV in the evaluation directory.
    episodes_csv = eval_dir / "episodes.csv"
    csv_files = [episodes_csv] + sorted(
        p for p in eval_dir.glob("*.csv") if p != episodes_csv
    )

    inspected = []

    for csv_path in csv_files:
        if not csv_path.exists():
            continue

        try:
            header_df = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue

        columns = list(header_df.columns)
        inspected.append((csv_path.name, columns))

        index_col = next((c for c in index_candidates if c in columns), None)
        success_col = next((c for c in success_candidates if c in columns), None)

        if index_col is None or success_col is None:
            continue

        full_df = pd.read_csv(csv_path, usecols=[index_col, success_col])
        success_bool = parse_bool_column(full_df[success_col])

        return dict(zip(full_df[index_col].astype(int), success_bool))

    details = "\n".join(f"  - {name}: {cols}" for name, cols in inspected)
    raise ValueError(
        "Could not locate lap success information (a lap-index column plus "
        f"a success column) in any CSV under {eval_dir}.\n"
        f"Inspected files/columns:\n{details or '  (no CSV files found)'}"
    )


def choose_tick_interval(max_step):
    """Pick a readable x-axis tick spacing for a given number of steps."""
    target_ticks = 10
    rough = max(max_step / target_ticks, 1)
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    return next((step for step in nice_steps if step >= rough), nice_steps[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-step frame acquisition for the first 10 evaluation laps."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Evaluation directory containing lap_00 .. lap_09 subfolders",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    # ---------------------------------------------------------
    # Load per-step frame-acquisition data (lap_00 .. lap_09 only)
    # ---------------------------------------------------------

    laps = load_laps(eval_dir)

    if not laps:
        raise ValueError(f"No lap_00 .. lap_09 steps.csv files found under {eval_dir}")

    lap_numbers = [lap_num for lap_num, _ in laps]
    n_laps = len(laps)
    max_step = max(int(df["step"].max()) for _, df in laps)

    # 1 = frame acquired, 0 = frame not acquired, NaN = episode ended / no data.
    matrix = np.full((n_laps, max_step + 1), np.nan)

    for row, (_, df) in enumerate(laps):
        steps = df["step"].to_numpy(dtype=int)
        matrix[row, steps] = df["frame_consumed"].to_numpy(dtype=float)

    # ---------------------------------------------------------
    # Load lap success from the episode-level summary
    # ---------------------------------------------------------

    success_lookup = find_success_lookup(eval_dir)

    success_values = []
    for lap_num in lap_numbers:
        if lap_num not in success_lookup:
            raise ValueError(
                f"No success value found for lap {lap_num} in the episode "
                f"summary under {eval_dir}."
            )
        success_values.append(bool(success_lookup[lap_num]))

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

    cmap = ListedColormap([NOT_ACQUIRED_COLOR, ACQUIRED_COLOR])
    cmap.set_bad(color=NO_DATA_COLOR)

    fig = plt.figure(figsize=(9.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[9, 1], wspace=0.06)

    ax_raster = fig.add_subplot(gs[0, 0])
    ax_success = fig.add_subplot(gs[0, 1], sharey=ax_raster)

    # ---------------------------------------------------------
    # Main raster: control step vs. lap
    # ---------------------------------------------------------

    ax_raster.imshow(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    ax_raster.set_xlabel("Time Step")
    ax_raster.set_ylabel("Episode")

    ax_raster.set_yticks(range(n_laps))
    ax_raster.set_yticklabels([str(lap_num) for lap_num in lap_numbers])

    ax_raster.xaxis.set_major_locator(MultipleLocator(choose_tick_interval(max_step)))
    ax_raster.set_xlim(-0.5, max_step + 0.5)

    for spine in ax_raster.spines.values():
        spine.set_linewidth(1.1)

    # ---------------------------------------------------------
    # Aligned success column (Yes / No text, not part of the time axis)
    # ---------------------------------------------------------

    ax_success.set_xlim(0, 1)
    ax_success.set_xticks([])
    ax_success.tick_params(axis="y", left=False, labelleft=False)
    ax_success.set_title("Success", fontsize=10, pad=6)

    for row, success in enumerate(success_values):
        ax_success.text(
            0.5,
            row,
            "Yes" if success else "No",
            ha="center",
            va="center",
            fontsize=9,
        )

    for spine in ax_success.spines.values():
        spine.set_linewidth(1.1)

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    title_lines = ["Frame Acquisition Across Evaluation Laps"]

    frame_cost, budget = parse_policy_name(eval_dir.name)
    if frame_cost is not None and budget is not None:
        fc_label = "0" if frame_cost == 0.0 else f"{frame_cost:g}"
        title_lines.append(f"$f_c={fc_label},\\ B={int(budget)}$")

    fig.suptitle("\n".join(title_lines), fontsize=13)

    # ---------------------------------------------------------
    # Legend (below the raster so it never covers plotted cells)
    # ---------------------------------------------------------

    legend_elements = [
        Patch(facecolor=ACQUIRED_COLOR, edgecolor="none", label="Frame acquired"),
        Patch(facecolor=NOT_ACQUIRED_COLOR, edgecolor="none", label="Frame not acquired"),
        Patch(facecolor=NO_DATA_COLOR, edgecolor="black", linewidth=0.5, label="Episode ended"),
    ]

    ax_raster.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.18),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    png_path = eval_dir / "frame_acquisition_episodes.png"
    pdf_path = eval_dir / "frame_acquisition_episodes.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Laps plotted: {n_laps}")
    print(f"Maximum control step: {max_step}")
    print()
    print("Saved figures:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
