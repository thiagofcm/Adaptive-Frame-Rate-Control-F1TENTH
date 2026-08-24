"""
Evaluation script for rule-based sensing baselines, run through the exact
same AdaptiveFPSEnv/reward/logging pipeline as evaluate_adaptive_fps.py.

Three policies choosing only between 5 Hz and 10 Hz:

  gt_curvature          - curve -> 10 Hz, straight -> 5 Hz
  gt_curvature_inverse  - curve -> 5 Hz,  straight -> 10 Hz
  random                - 5/10 Hz with probability 0.5 each, --seed for reproducibility

"Curve" vs "straight" is decided from ground-truth track geometry (the
raceline's own precomputed curvature column, maps/<map>_raceline.csv), not
from steering state or commanded steering.

This script imports run_lap/summarize_results/FPS_TO_ACTION/etc. directly
from evaluate_adaptive_fps.py rather than reimplementing them, so the two
scripts' episodes.csv/steps.csv/trajectory_lap_XX.npy outputs stay
byte-for-byte schema-identical - AdaptiveFPS/utils/compute_statistics.py,
AdaptiveFPS/scripts/analyze_adaptive_fps.py, and
AdaptiveFPS/utils/plot_frame_acquisition.py all work on these baseline
outputs unmodified. evaluate_adaptive_fps.py and AdaptiveFPSEnv itself are
never touched.

Run from the repo root:
    python -m AdaptiveFPS.scripts.evaluate_baselines \\
        --policy gt_curvature --map f1_aut --n-laps 100 --curvature-threshold 0.1

    python -m AdaptiveFPS.scripts.evaluate_baselines \\
        --policy random --map f1_aut --n-laps 100 --seed 0
"""
import argparse
import atexit
import os

import numpy as np

from AdaptiveFPS.scripts.evaluate_adaptive_fps import (
    FPS_TO_ACTION, EVAL_ROOT, EPISODE_CSV_FIELDS, MODEL_RUN_NAME_DEFAULT,
    MAP_NAME_DEFAULT, RUN_FILE, run_lap, summarize_results,
)
from AdaptiveFPS.envs.adaptive_fps_env import AdaptiveFPSEnv

import csv

DEFAULT_CURVATURE_THRESHOLD = 0.1  # rad/m; see module docstring / plan for how this was picked


class RacelineCurvature:
    """Ground-truth track curvature, read straight from the raceline CSV's
    own precomputed kappa column (maps/<map>_raceline.csv, standard TUM
    format: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2) - no
    curve locations are hardcoded, this works for any track that has a
    raceline file."""

    def __init__(self, map_name):
        data = np.loadtxt(f"maps/{map_name}_raceline.csv", delimiter=",")
        self.wpts = data[:, 1:3]   # x, y
        self.kappas = data[:, 4]   # kappa_radpm

    def curvature_at(self, x, y):
        dists = np.linalg.norm(self.wpts - np.array([x, y]), axis=1)
        return self.kappas[np.argmin(dists)]


class CurvatureBaselinePolicy:
    """Chooses 5/10 Hz from ground-truth curvature at the vehicle's current
    position. Exposes the same .predict(observation, lstm_state, done,
    deterministic) -> (action, lstm_state) interface run_lap() calls on its
    `model` argument, so run_lap() itself needs no changes."""

    def __init__(self, env, curvature_lookup, threshold, invert):
        self.env = env
        self.curvature_lookup = curvature_lookup
        self.threshold = threshold
        self.invert = invert
        self.n_straight = 0
        self.n_curve = 0
        self.n_5hz = 0
        self.n_10hz = 0

    def predict(self, observation, lstm_state, done, deterministic=True):
        # Ground-truth vehicle position - env.current_observation["state"] is
        # rebuilt from the raw simulator pose every control step regardless
        # of the sensing decision (see AdaptiveFPSEnv._build_observation),
        # so this is never stale.
        x, y = self.env.current_observation["state"][0], self.env.current_observation["state"][1]
        is_curve = bool(abs(self.curvature_lookup.curvature_at(x, y)) >= self.threshold)

        if is_curve:
            self.n_curve += 1
        else:
            self.n_straight += 1

        if self.invert:
            fps = 5 if is_curve else 10
        else:
            fps = 10 if is_curve else 5

        if fps == 10:
            self.n_10hz += 1
        else:
            self.n_5hz += 1

        return FPS_TO_ACTION[fps], lstm_state

    def print_debug_stats(self):
        n_total = self.n_straight + self.n_curve
        if n_total == 0:
            return
        print(f"  straight={100.0 * self.n_straight / n_total:.1f}%  curve={100.0 * self.n_curve / n_total:.1f}%   "
              f"5Hz={100.0 * self.n_5hz / n_total:.1f}%  10Hz={100.0 * self.n_10hz / n_total:.1f}%")


class RandomBaselinePolicy:
    """Chooses 5/10 Hz uniformly at random at every control step."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.n_5hz = 0
        self.n_10hz = 0

    def predict(self, observation, lstm_state, done, deterministic=True):
        fps = int(self.rng.choice([5, 10]))
        if fps == 10:
            self.n_10hz += 1
        else:
            self.n_5hz += 1
        return FPS_TO_ACTION[fps], lstm_state

    def print_debug_stats(self):
        n_total = self.n_5hz + self.n_10hz
        if n_total == 0:
            return
        print(f"  5Hz={100.0 * self.n_5hz / n_total:.1f}%  10Hz={100.0 * self.n_10hz / n_total:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, choices=["gt_curvature", "gt_curvature_inverse", "random"])
    parser.add_argument("--map", default=MAP_NAME_DEFAULT)
    parser.add_argument("--n-laps", type=int, default=100)
    parser.add_argument("--nav-model", default=MODEL_RUN_NAME_DEFAULT)
    parser.add_argument("--curvature-threshold", type=float, default=DEFAULT_CURVATURE_THRESHOLD,
                         help="abs(curvature) >= this (rad/m) classifies a control step as a curve; "
                              "only used by gt_curvature/gt_curvature_inverse")
    parser.add_argument("--seed", type=int, default=0, help="only used by --policy random")
    parser.add_argument("--fc", type=float, default=0.0, help="frame cost passed through to AdaptiveFPSEnv (reward calculation unchanged)")
    parser.add_argument("--bud", type=float, default=300.0, help="budget passed through to AdaptiveFPSEnv (reward calculation unchanged)")
    parser.add_argument("--budget-penalty", type=float, default=10.0, help="budget penalty passed through to AdaptiveFPSEnv (reward calculation unchanged)")
    args = parser.parse_args()

    atexit.register(summarize_results, args.map)

    env = AdaptiveFPSEnv(RUN_FILE, args.map, args.nav_model, args.bud, args.fc, args.budget_penalty)

    if args.policy == "gt_curvature":
        policy = CurvatureBaselinePolicy(env, RacelineCurvature(args.map), args.curvature_threshold, invert=False)
        out_dir = f"{EVAL_ROOT}/{args.map}/gt_curvature"
    elif args.policy == "gt_curvature_inverse":
        policy = CurvatureBaselinePolicy(env, RacelineCurvature(args.map), args.curvature_threshold, invert=True)
        out_dir = f"{EVAL_ROOT}/{args.map}/gt_curvature_inverse"
    else:
        policy = RandomBaselinePolicy(args.seed)
        out_dir = f"{EVAL_ROOT}/{args.map}/random_5_10Hz_seed_{args.seed}"

    model_label = args.policy

    episode_rows = []
    for lap_index in range(args.n_laps):
        lap_dir = f"{out_dir}/lap_{lap_index:02d}"
        row = run_lap(env, None, lap_dir, policy, lap_index)
        row["lap_index"] = lap_index
        episode_rows.append(row)

        print(f"[{model_label}] Lap {lap_index}: success={row['success']} crashed={row['crashed']} "
              f"lap_time={row['lap_time']:.2f} progress={row['final_progress']:.3f} "
              f"adaptive_return={row['episode_return']:.3f} nav_return={row['nav_episode_return']:.3f} ",
              f"mean_fps={row['mean_fps']:.2f} ", f"fresh_obs={row['n_fresh_observations']}/{row['steps_length']} "
              f"({row['fresh_observation_ratio']:.2f})")

    env.close()

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/episodes.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Episode results written to: {out_dir}/episodes.csv")

    # ---------------------------------
    # End-of-run summary (console only)
    # ---------------------------------
    n_laps = len(episode_rows)
    successes = [row for row in episode_rows if row["success"]]
    crashes = [row for row in episode_rows if row["crashed"]]
    success_rate = 100.0 * len(successes) / n_laps if n_laps else 0.0
    crash_rate = 100.0 * len(crashes) / n_laps if n_laps else 0.0

    print("_________________________________________________________")
    print(f"[{model_label}] Summary over {n_laps} laps:")
    print(f"  success_rate={success_rate:.2f}%  crash_rate={crash_rate:.2f}%")
    if successes:
        successful_times = np.array([row["lap_time"] for row in successes])
        print(f"  successful lap_time: mean={successful_times.mean():.4f} std={successful_times.std():.4f}")
    print(f"  mean final_progress={np.mean([row['final_progress'] for row in episode_rows]):.4f}")
    print(f"  mean adaptive_episode_return=" f"{np.mean([row['episode_return'] for row in episode_rows]):.4f}")
    print(f"  mean nav_episode_return="f"{np.mean([row['nav_episode_return'] for row in episode_rows]):.4f}")
    print(f"  mean n_fresh_observations={np.mean([row['n_fresh_observations'] for row in episode_rows]):.4f}")
    print(f"  mean steps={np.mean([row['steps_length'] for row in episode_rows]):.4f}")
    print(f"  mean fresh_observation_ratio={np.mean([row['fresh_observation_ratio'] for row in episode_rows]):.4f}")

    print("_________________________________________________________")
    print(f"[{model_label}] Sensing-decision breakdown over {n_laps} laps:")
    policy.print_debug_stats()


if __name__ == '__main__':
    main()
