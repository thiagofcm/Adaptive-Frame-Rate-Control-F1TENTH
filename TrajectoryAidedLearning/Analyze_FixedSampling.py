"""
Analysis/plotting for TestSimulation_FixedSampling.py results.

This script never touches the simulator (no gym/f110_env/AgentTester/torch
imports) - it only loads the episodes.csv / steps.csv / Lap_*_history_*.npy
files that TestSimulation_FixedSampling.py already saved under:

    Data/FixedSamplingResults/<model>/FixedSampling/interval_<N>/episodes.csv
    Data/FixedSamplingResults/<model>/FixedSampling/interval_<N>/lap_<i>/steps.csv
    Data/FixedSamplingResults/<model>/FixedSampling/interval_<N>/lap_<i>/Lap_<i>_history_..._<map>.npy

Usage examples (run from the repo root):

    # plot track + velocity profile + sense_interval label for one saved trajectory
    python -m TrajectoryAidedLearning.Analyze_FixedSampling \\
        --npy Data/FixedSamplingResults/fast_..._6_5_0/FixedSampling/interval_2/lap_03/Lap_3_history_..._f1_aut.npy

    # overlay every lap of an interval, successes vs failures, crash locations marked
    python -m TrajectoryAidedLearning.Analyze_FixedSampling --interval 2 --overlay

    # same, but only laps 3 and 11
    python -m TrajectoryAidedLearning.Analyze_FixedSampling --interval 2 --overlay --laps 3 11

    # progress / instantaneous reward / cumulative reward over time for laps 3 and 11
    python -m TrajectoryAidedLearning.Analyze_FixedSampling --interval 2 --timeseries --laps 3 11
"""

import argparse
import csv
import glob
import os
import re

import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt


from TrajectoryAidedLearning.Utils.StdTrack import StdTrack

DEFAULT_MODEL_RUN_NAME = "fast_Std_Std_TAL_f1_aut_6_5_0"
DEFAULT_MAP_NAME = "f1_aut"

# full_states + action columns, exactly as written by VehicleStateHistory.save_history():
# RaceCar.state is [x, y, steering_angle, velocity, yaw, yaw_rate, slip_angle],
# followed by the 2 commanded action columns [steering, speed].
NPY_COL_X, NPY_COL_Y, NPY_COL_STEER_STATE, NPY_COL_V, NPY_COL_YAW = 0, 1, 2, 3, 4
NPY_COL_CMD_STEER, NPY_COL_CMD_SPEED = 7, 8


def fixed_sampling_root(model_run_name):
    return f"Data/FixedSamplingResults/{model_run_name}/FixedSampling"


def interval_dir(model_run_name, sense_interval):
    return f"{fixed_sampling_root(model_run_name)}/interval_{sense_interval}"


def parse_interval_from_path(path):
    match = re.search(r"interval_(\d+)", path)
    return int(match.group(1)) if match else None


def parse_lap_index_from_path(path):
    match = re.search(r"lap_(\d+)", path)
    return int(match.group(1)) if match else None


def build_output_filename(map_name, model_run_name, sense_interval, plot_type, extra=None):
    """e.g. f1_aut_fast_Std_Std_TAL_f1_aut_6_5_0_interval2_overlay_laps3-11.png"""
    interval_part = f"interval{sense_interval}" if sense_interval is not None else "intervalNA"
    parts = [map_name, model_run_name, interval_part, plot_type]
    if extra:
        parts.append(extra)
    return "_".join(parts) + ".png"


def save_figure(fig, output_dir, map_name, model_run_name, sense_interval, plot_type, extra=None):
    filename = build_output_filename(map_name, model_run_name, sense_interval, plot_type, extra)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def load_episode_rows(model_run_name, sense_interval):
    path = f"{interval_dir(model_run_name, sense_interval)}/episodes.csv"
    with open(path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row['lap_index'] = int(row['lap_index'])
        row['success'] = row['success'] == 'True'
        row['crashed'] = row['crashed'] == 'True'
        for key in ('lap_time', 'final_progress', 'episode_return', 'final_x', 'final_y', 'fresh_observation_ratio'):
            row[key] = float(row[key])

    return sorted(rows, key=lambda row: row['lap_index'])


def load_step_rows(model_run_name, sense_interval, lap_index):
    path = f"{interval_dir(model_run_name, sense_interval)}/lap_{lap_index:02d}/steps.csv"
    with open(path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row['control_step'] = int(row['control_step'])
        row['lidar_fresh'] = row['lidar_fresh'] == 'True'
        for key in ('lap_time', 'x', 'y', 'yaw', 'velocity', 'steering_state',
                    'commanded_steering', 'commanded_speed', 'instantaneous_reward',
                    'cumulative_reward', 'progress'):
            row[key] = float(row[key])

    return rows


def find_trajectory_npy(model_run_name, sense_interval, lap_index):
    pattern = f"{interval_dir(model_run_name, sense_interval)}/lap_{lap_index:02d}/Lap_*_history_*.npy"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No trajectory .npy found for {pattern}")
    return matches[0]


def load_trajectory_npy(path):
    data = np.load(path)
    return {
        "x": data[:, NPY_COL_X],
        "y": data[:, NPY_COL_Y],
        "yaw": data[:, NPY_COL_YAW],
        "velocity": data[:, NPY_COL_V],
        "steering_state": data[:, NPY_COL_STEER_STATE],
        "commanded_steering": data[:, NPY_COL_CMD_STEER],
        "commanded_speed": data[:, NPY_COL_CMD_SPEED],
    }


def load_map_image(map_name):
    """The actual map raster + its world-frame extent, in the same convention
    laser_models.py's ScanSimulator2D.set_map() uses for LiDAR raycasting:
    resolution/origin come from maps/<map>.yaml, and the raw (unflipped) PNG's
    row 0 is the top of the map = the highest y - which is exactly what
    matplotlib's imshow(..., origin='upper') expects, so no flip is needed
    here (set_map() flips it the other way only because its own raycasting
    code indexes rows bottom-up)."""
    with open(f"maps/{map_name}.yaml", 'r') as file:
        map_meta = yaml.safe_load(file)

    image = np.array(Image.open(f"maps/{map_meta['image']}").convert('L'))
    resolution = map_meta['resolution']
    origin_x, origin_y = map_meta['origin'][0], map_meta['origin'][1]
    height, width = image.shape

    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]
    return image, extent


def draw_track(ax, map_name):
    image, extent = load_map_image(map_name)
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=extent, origin='upper', zorder=0)

    track = StdTrack(map_name)
    ax.plot(track.wpts[:, 0], track.wpts[:, 1], '--', color='tab:orange', linewidth=1, zorder=1, label='centerline')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


def find_lap_time_for_npy(npy_path):
    """Look up this lap's total time from the steps.csv saved alongside the
    .npy (same lap_<i>/ directory) - the .npy itself has no time column."""
    steps_path = os.path.join(os.path.dirname(npy_path), "steps.csv")
    if not os.path.exists(steps_path):
        return None

    with open(steps_path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    return float(rows[-1]['lap_time']) if rows else None


def plot_single_trajectory(npy_path, map_name, sense_interval=None):
    if sense_interval is None:
        sense_interval = parse_interval_from_path(npy_path)
    lap_time = find_lap_time_for_npy(npy_path)

    trajectory = load_trajectory_npy(npy_path)

    fig, (ax_track, ax_vel) = plt.subplots(1, 2, figsize=(14, 6))

    draw_track(ax_track, map_name)
    scatter = ax_track.scatter(trajectory['x'], trajectory['y'], c=trajectory['velocity'],
                                cmap='viridis', s=8, zorder=1)
    fig.colorbar(scatter, ax=ax_track, label='velocity [m/s]')
    ax_track.set_title(f"Trajectory: {os.path.basename(npy_path)}")

    ax_vel.plot(trajectory['velocity'])
    ax_vel.set_xlabel('control step')
    ax_vel.set_ylabel('velocity [m/s]')
    ax_vel.set_title('Velocity profile')

    interval_label = f"sense_interval = {sense_interval}" if sense_interval is not None else "sense_interval = unknown"
    time_label = f"time = {lap_time:.2f}s" if lap_time is not None else "time = unknown"
    fig.suptitle(f"{interval_label}   |   {time_label}")
    fig.tight_layout()

    return fig


def plot_overlay(model_run_name, map_name, sense_interval, lap_indices=None):
    episode_rows = load_episode_rows(model_run_name, sense_interval)
    if lap_indices is not None:
        episode_rows = [row for row in episode_rows if row['lap_index'] in lap_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, map_name)

    success_labelled, failure_labelled = False, False
    for row in episode_rows:
        steps = load_step_rows(model_run_name, sense_interval, row['lap_index'])
        xs = [step['x'] for step in steps]
        ys = [step['y'] for step in steps]

        if row['success']:
            ax.plot(xs, ys, color='tab:green', alpha=0.6, linewidth=1,
                     label='successful' if not success_labelled else None)
            success_labelled = True
        else:
            ax.plot(xs, ys, color='tab:red', alpha=0.6, linewidth=1,
                     label='failed' if not failure_labelled else None)
            ax.plot(row['final_x'], row['final_y'], 'x', color='black', markersize=8, zorder=2)
            failure_labelled = True

        # time required for this trajectory, next to where it ends
        ax.annotate(f"{row['lap_time']:.1f}s", (xs[-1], ys[-1]), fontsize=6,
                    color='0.2', xytext=(3, 3), textcoords='offset points')

    ax.legend()
    ax.set_title(f"{model_run_name} on {map_name} - sense_interval = {sense_interval}\n"
                 f"{sum(r['success'] for r in episode_rows)}/{len(episode_rows)} laps successful")
    fig.tight_layout()

    return fig


def plot_timeseries(model_run_name, sense_interval, lap_indices):
    lap_times = {row['lap_index']: row['lap_time'] for row in load_episode_rows(model_run_name, sense_interval)}

    fig, (ax_progress, ax_reward, ax_return) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for lap_index in lap_indices:
        steps = load_step_rows(model_run_name, sense_interval, lap_index)
        control_steps = [step['control_step'] for step in steps]
        label = f"lap {lap_index} ({lap_times[lap_index]:.1f}s)" if lap_index in lap_times else f"lap {lap_index}"

        ax_progress.plot(control_steps, [step['progress'] for step in steps], label=label)
        ax_reward.plot(control_steps, [step['instantaneous_reward'] for step in steps], label=label)
        ax_return.plot(control_steps, [step['cumulative_reward'] for step in steps], label=label)

    ax_progress.set_ylabel('progress')
    ax_reward.set_ylabel('instantaneous reward')
    ax_return.set_ylabel('cumulative reward')
    ax_return.set_xlabel('control step')
    ax_progress.legend()
    ax_progress.set_title(f"{model_run_name} - sense_interval = {sense_interval}")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_RUN_NAME, help="model run_name")
    parser.add_argument("--map", default=DEFAULT_MAP_NAME, help="map name (for drawing the track)")
    parser.add_argument("--interval", type=int, default=None, help="sense_interval to load (for --overlay/--timeseries)")
    parser.add_argument("--laps", type=int, nargs='+', default=None, help="restrict to these lap indices")
    parser.add_argument("--npy", default=None, help="plot one saved trajectory .npy file (track + velocity)")
    parser.add_argument("--overlay", action='store_true', help="overlay successful/failed trajectories for --interval")
    parser.add_argument("--timeseries", action='store_true', help="plot progress/reward/return over time for --laps")
    parser.add_argument("--output-dir",default="Data/FixedSamplingAnalysis",help="directory where plots are saved")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not (args.npy or args.overlay or args.timeseries):
        parser.error("nothing to do: pass --npy, --overlay, or --timeseries")

    if args.npy:
        sense_interval = args.interval if args.interval is not None else parse_interval_from_path(args.npy)
        lap_index = parse_lap_index_from_path(args.npy)
        fig = plot_single_trajectory(args.npy, args.map, sense_interval=sense_interval)
        extra = f"lap{lap_index}" if lap_index is not None else None
        save_figure(fig, args.output_dir, args.map, args.model, sense_interval, "trajectory", extra)

    if args.overlay:
        if args.interval is None:
            parser.error("--overlay requires --interval")
        fig = plot_overlay(args.model, args.map, args.interval, lap_indices=args.laps)
        extra = "laps" + "-".join(str(lap) for lap in args.laps) if args.laps else None
        save_figure(fig, args.output_dir, args.map, args.model, args.interval, "overlay", extra)

    if args.timeseries:
        if args.interval is None:
            parser.error("--timeseries requires --interval")
        laps = args.laps
        if laps is None:
            episode_rows = load_episode_rows(args.model, args.interval)
            laps = [row['lap_index'] for row in episode_rows]
        fig = plot_timeseries(args.model, args.interval, laps)
        extra = "laps" + "-".join(str(lap) for lap in laps)
        save_figure(fig, args.output_dir, args.map, args.model, args.interval, "timeseries", extra)


if __name__ == '__main__':
    main()
