"""
compute_statistics.py
──────────────────────
Two modes:

1. Single evaluation mode — computes sensing-efficiency statistics from an
   episode-level evaluation CSV and writes a one-row statistics.csv
   alongside it.

       python compute_statistics.py --csv path/to/episodes.csv

2. Aggregation mode — walks an evaluation root directory, collects every
   nested statistics.csv (one per policy subfolder), and combines them into
   a single sorted statistics_summary.csv in the root.

       python compute_statistics.py --aggregate-root path/to/eval_root
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_success_column(series: pd.Series) -> pd.Series:
    """Robustly convert a success column to bool, handling True/False,
    "true"/"false" (any case), and 1/0 (numeric or string)."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def compute_statistics(csv_path: Path):
    df = pd.read_csv(csv_path)

    steps_length = df["steps_length"]
    n_fresh = df["n_fresh_observations"]

    # ── 1. Measurement : No-Measurement ratio ──────────────────────
    # Computed per episode so we can report the distribution (mean/std),
    # not just an aggregate ratio. Episodes with zero fresh observations
    # would divide by zero, so those are excluded via NaN + nanmean/nanstd.
    no_measurements = steps_length - n_fresh
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = no_measurements / n_fresh
    ratio = ratio.where(n_fresh != 0, np.nan)
    mean_ratio = np.nanmean(ratio)
    std_ratio = np.nanstd(ratio)

    # ── 2. Observation reduction (%) ────────────────────────────────
    # Per-episode percentage of steps without a fresh observation, then
    # aggregated — deliberately not sum(n_fresh)/sum(steps_length), since
    # that would hide the per-episode distribution.
    with np.errstate(divide="ignore", invalid="ignore"):
        observation_reduction = (1 - n_fresh / steps_length) * 100
    observation_reduction = observation_reduction.where(steps_length != 0, np.nan)
    mean_reduction = np.nanmean(observation_reduction)
    std_reduction = np.nanstd(observation_reduction)

    # ── 3. Fresh observations | success ─────────────────────────────
    success = parse_success_column(df["success"])
    successful_fresh = n_fresh[success]
    n_success = len(successful_fresh)
    if n_success > 0:
        mean_fresh_success = successful_fresh.mean()
        std_fresh_success = successful_fresh.std()
    else:
        mean_fresh_success = float("nan")
        std_fresh_success = float("nan")

    # ── Write statistics.csv ────────────────────────────────────────
    out_path = csv_path.parent / "statistics.csv"
    summary = pd.DataFrame([{
        "mean_measurement_no_measurement_ratio": round(mean_ratio, 4),
        "std_measurement_no_measurement_ratio": round(std_ratio, 4),
        "mean_observation_reduction_percent": round(mean_reduction, 4),
        "std_observation_reduction_percent": round(std_reduction, 4),
        "mean_fresh_observations_given_success": round(mean_fresh_success, 4),
        "std_fresh_observations_given_success": round(std_fresh_success, 4),
    }])
    # na_rep="NaN" so a fully-empty-success eval writes literal "NaN", not a blank cell
    summary.to_csv(out_path, index=False, na_rep="NaN")

    # ── Console summary ─────────────────────────────────────────────
    print(f"Processed {len(df)} episodes")
    print(f"Successful episodes: {n_success}")
    print()
    print(f"Measurement : No Measurement = 1:({mean_ratio:.2f} ± {std_ratio:.2f})")
    print(f"Observation Reduction = {mean_reduction:.2f} ± {std_reduction:.2f}%")
    print(f"Fresh Observations | Success = {mean_fresh_success:.2f} ± {std_fresh_success:.2f}")
    print()
    print("Saved statistics to:")
    print(out_path)


def policy_sort_key(policy: str):
    """Fixed policies (ascending Hz) sort before adaptive policies (ascending
    frame cost, then descending budget); unparsable names sort last, alphabetically."""
    fixed_match = re.match(r"^fixed_(\d+(?:\.\d+)?)Hz$", policy)
    if fixed_match:
        return (0, float(fixed_match.group(1)), 0.0, policy)

    # Unanchored at the end so an optional trailing "_bp_<budget_penalty>" (or nothing) both match.
    adaptive_match = re.match(r"^adaptive_fc_([\d.]+)_bud_([\d.]+)", policy)
    if adaptive_match:
        frame_cost = float(adaptive_match.group(1))
        budget = float(adaptive_match.group(2))
        return (1, frame_cost, -budget, policy)

    return (2, 0.0, 0.0, policy)


def aggregate_statistics(root_path: Path):
    if not root_path.exists():
        print(f"Error: aggregate root does not exist: {root_path}", file=sys.stderr)
        sys.exit(1)

    # Exact filename match only, so statistics_summary.csv (this script's own
    # output) and unrelated files (e.g. summary.csv) are never picked up.
    stats_files = sorted(root_path.rglob("statistics.csv"))
    print(f"Found {len(stats_files)} statistics.csv files")

    if not stats_files:
        print("No statistics.csv files found; nothing to aggregate.")
        return

    rows = []
    skipped = 0
    for stats_file in stats_files:
        try:
            df = pd.read_csv(stats_file)
        except pd.errors.EmptyDataError:
            print(f"Warning: {stats_file} is empty, skipping")
            skipped += 1
            continue
        if df.empty:
            print(f"Warning: {stats_file} has no rows, skipping")
            skipped += 1
            continue
        # Scalar assignment broadcasts to every row, so a statistics.csv with
        # more than one row is supported rather than silently discarded.
        df.insert(0, "policy", stats_file.parent.name)
        rows.append(df)

    if not rows:
        print("No usable statistics.csv files found; nothing to aggregate.")
        return

    # sort=False keeps column order stable and lets differing schemas union
    # together, with missing metrics becoming NaN rather than raising.
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["_sort_key"] = combined["policy"].map(policy_sort_key)
    combined = (
        combined.sort_values("_sort_key", kind="stable")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    out_path = root_path / "statistics_summary.csv"
    combined.to_csv(out_path, index=False, na_rep="NaN")

    print(f"Aggregated {len(rows)} policy evaluations")
    if skipped:
        print(f"Skipped {skipped} statistics.csv file(s)")
    print()
    print("Saved aggregated statistics to:")
    print(out_path)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, default=None, help="Path to evaluation CSV")
    group.add_argument(
        "--aggregate-root",
        type=str,
        default=None,
        help="Root evaluation directory containing policy subfolders with statistics.csv files",
    )
    args = parser.parse_args()

    if args.csv:
        compute_statistics(Path(args.csv))
    else:
        aggregate_statistics(Path(args.aggregate_root))


if __name__ == "__main__":
    main()
