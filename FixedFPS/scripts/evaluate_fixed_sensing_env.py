"""
Evaluation script for FixedSensingF110Env - owns the fixed-sensing
experiment: the explicit temporal sensing gate lives HERE, in the episode
loop, not inside the env. This is deliberate: this exact seam is where a
later adaptive-FPS policy will replace the fixed modulo rule.

Run from the repo root:
    python -m FixedFPS.scripts.evaluate_fixed_sensing_env --sense-interval 2
"""
import argparse
import atexit
import csv
import glob
import os

import numpy as np

from FixedFPS.envs.fixed_sensing_f110_env import FixedSensingF110Env

MODEL_RUN_NAME_DEFAULT = "fast_Std_Std_TAL_f1_aut_6_5_0"
MAP_NAME_DEFAULT = "f1_aut"
RUN_FILE = "TAL_maps"
PHYSICS_TIMESTEP = 0.01  # matches F110Env/RaceCar's hardcoded default

EVAL_ROOT = "FixedFPS/eval"
EPISODE_CSV_FIELDS = [
    "sense_interval", "effective_sensing_hz", "lap_index", "success", "crashed",
    "lap_time", "final_progress", "episode_return", "n_control_steps",
    "n_fresh_observations", "fresh_observation_ratio", "final_x", "final_y",
]

SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_CSV_FIELDS = [
    "sense_interval", "n_laps", "success_rate", "mean_lap_time", "std_lap_time",
    "mean_episode_return", "std_episode_return", "mean_final_progress",
]


def summarize_results(map_name):
    """Read every interval_*/episodes.csv under FixedFPS/eval/<map_name>/ and
    summarize mean/std lap_time, episode_return, final_progress and
    success_rate per sense_interval, across however many laps are recorded
    for it.

    Registered to run on interpreter exit so it always reflects the latest
    accumulated results, whether the run finished normally or was interrupted.
    """
    map_dir = f"{EVAL_ROOT}/{map_name}"
    episode_csv_paths = sorted(glob.glob(f"{map_dir}/interval_*/episodes.csv"))
    if not episode_csv_paths:
        return

    summary_rows = []
    for path in episode_csv_paths:
        with open(path, 'r', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue

        lap_times = np.array([float(row['lap_time']) for row in rows])
        successes = np.array([row['success'] == 'True' for row in rows]) # each lap's success as a bool
        returns = np.array([float(row['episode_return']) for row in rows])
        final_progress = np.array([float(row['final_progress']) for row in rows])

        summary_rows.append({
            "sense_interval": int(rows[0]['sense_interval']),
            "n_laps": len(rows),
            "success_rate": float(successes.mean() * 100),
            "mean_lap_time": float(lap_times.mean()),
            "std_lap_time": float(lap_times.std()),
            "mean_episode_return": float(returns.mean()),
            "std_episode_return": float(returns.std()),
            "mean_final_progress": float(final_progress.mean()),
        })

    summary_rows.sort(key=lambda row: row['sense_interval'])

    summary_path = f"{map_dir}/{SUMMARY_CSV_NAME}"
    with open(summary_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("_________________________________________________________")
    print(f"Sense-interval summary for map '{map_name}' (from all interval_*/episodes.csv found on disk):")
    for row in summary_rows:
        print(f"  interval={row['sense_interval']:>3}  n_laps={row['n_laps']:>3}  "
              f"success_rate={row['success_rate']:6.2f}%  "
              f"lap_time mean={row['mean_lap_time']:.4f}  std={row['std_lap_time']:.4f}  "
              f"return mean={row['mean_episode_return']:.4f}")
    print(f"Summary written to: {summary_path}")


def run_lap(env, sense_interval):
    observation, info = env.reset()

    control_step_count = 0
    last_fresh_scan = None
    n_fresh_observations = 0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # ---------------------------------
        # 1. Explicit temporal sensing gate
        # ---------------------------------
        lidar_fresh = (control_step_count % sense_interval == 0)

        if lidar_fresh:
            last_fresh_scan = observation["scan"].copy()
            n_fresh_observations += 1

        control_step_count += 1

        # ---------------------------------
        # 2. Environment/control transition
        # ---------------------------------
        observation, nav_reward, terminated, truncated, info = env.control_step(last_fresh_scan)

    return {
        "success": observation["lap_done"],
        "crashed": observation["colision_done"],
        "lap_time": observation["current_laptime"],
        "final_progress": info["progress"],
        "episode_return": info["cumulative_reward"],
        "n_control_steps": control_step_count,
        "n_fresh_observations": n_fresh_observations,
        "fresh_observation_ratio": n_fresh_observations / control_step_count if control_step_count else 0.0,
        "final_x": observation["state"][0],
        "final_y": observation["state"][1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sense-interval", type=int, required=True)
    parser.add_argument("--n-laps", type=int, default=20)
    parser.add_argument("--model", default=MODEL_RUN_NAME_DEFAULT)
    parser.add_argument("--map", default=MAP_NAME_DEFAULT)
    args = parser.parse_args()

    atexit.register(summarize_results, args.map)

    env = FixedSensingF110Env(RUN_FILE, args.map, args.model)
    effective_sensing_hz = (1.0 / (PHYSICS_TIMESTEP * env.conf.sim_steps)) / args.sense_interval

    episode_rows = []
    for lap_index in range(args.n_laps):
        row = run_lap(env, args.sense_interval)
        row.update({"sense_interval": args.sense_interval, "effective_sensing_hz": effective_sensing_hz, "lap_index": lap_index})
        episode_rows.append(row)
        print(f"Lap {lap_index}: success={row['success']} crashed={row['crashed']} "
              f"lap_time={row['lap_time']:.2f} return={row['episode_return']:.3f}")

    env.close()

    out_dir = f"{EVAL_ROOT}/{args.map}/interval_{args.sense_interval}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/episodes.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Episode results written to: {out_dir}/episodes.csv")


if __name__ == '__main__':
    main()
