"""
Analysis/plotting for AdaptiveFPS/scripts/evaluate_adaptive_fps.py results -
and, since both pipelines now write the same episodes.csv/lap_<i>/steps.csv
shape, this also reads FixedFPS/scripts/evaluate_fixed_sensing_env.py output
directly. Never touches the simulator (no gym/f110_env/AgentTester/torch
imports) and doesn't import Analyze_FixedSampling.py or FixedFPS/ - purely a
standalone reader of whatever CSV/.npy files are already on disk.

Directory-driven rather than model+interval-driven: point --eval-dir at
either pipeline's lap-data folder directly, e.g.

    AdaptiveFPS/eval/f1_aut/fixed_5Hz
    FixedFPS/eval/f1_aut/interval_2

Usage examples (run from the repo root):

    # plot track + velocity/FPS-colored trajectory for one saved lap
    python -m AdaptiveFPS.scripts.analyze_adaptive_fps \\
        --eval-dir AdaptiveFPS/eval/f1_aut/fixed_5Hz --npy-lap 3

    # overlay every lap, successes vs failures, crash locations marked
    python -m AdaptiveFPS.scripts.analyze_adaptive_fps \\
        --eval-dir AdaptiveFPS/eval/f1_aut/fixed_5Hz --overlay

    # progress / instantaneous reward / cumulative reward / FPS over time
    python -m AdaptiveFPS.scripts.analyze_adaptive_fps \\
        --eval-dir AdaptiveFPS/eval/f1_aut/fixed_5Hz --timeseries --laps 3 11

    # same reader, pointed at the older FixedFPS pipeline's output instead
    python -m AdaptiveFPS.scripts.analyze_adaptive_fps \\
        --eval-dir FixedFPS/eval/f1_aut/interval_2 --overlay
"""

import argparse
import csv
import glob
import os

import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm

from TrajectoryAidedLearning.Utils.StdTrack import StdTrack

FPS_CHOICES = [1, 2, 5, 10]  # the only valid discrete sensing-FPS actions

DEFAULT_MAP_NAME = "f1_aut"

# Two possible trajectory .npy layouts, told apart by column count:
#  - 9 cols: FixedFPS/TestSimulation_FixedFPS_Refactor.py's VehicleStateHistory
#    format - full_states (RaceCar.state: x,y,steering_angle,velocity,yaw,
#    yaw_rate,slip_angle) + 2 action columns.
#  - 8 cols: AdaptiveFPS/scripts/evaluate_adaptive_fps.py's own format -
#    AdaptiveFPSEnv never exposes full_states, so this uses the TAL 5-element
#    state (x,y,yaw,velocity,steering) + 2 action columns + current_fps.
NPY_COLS_9 = {"x": 0, "y": 1, "steering_state": 2, "velocity": 3, "yaw": 4, "commanded_steering": 7, "commanded_speed": 8}
NPY_COLS_8 = {"x": 0, "y": 1, "yaw": 2, "velocity": 3, "steering_state": 4, "commanded_steering": 5, "commanded_speed": 6, "current_fps": 7}


def build_output_filename(map_name, run_label, plot_type, extra=None):
    parts = [map_name, run_label, plot_type]
    if extra:
        parts.append(extra)
    return "_".join(parts) + ".png"


def save_figure(fig, output_dir, map_name, run_label, plot_type, extra=None):
    filename = build_output_filename(map_name, run_label, plot_type, extra)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def run_label_for(eval_dir):
    """A short label for filenames/titles, e.g. 'fixed_5Hz' or 'interval_2'."""
    return os.path.basename(eval_dir.rstrip('/'))


def describe_run(episode_rows):
    """Human-readable identifier for whichever schema this eval_dir's
    episodes.csv uses - 'mean_fps=X' (AdaptiveFPS) or 'sense_interval=N'
    (FixedFPS)."""
    if not episode_rows:
        return "unknown"
    row = episode_rows[0]
    if 'mean_fps' in row:
        return f"mean_fps={row['mean_fps']:.2f}"
    if 'sense_interval' in row:
        return f"sense_interval={row['sense_interval']}"
    return "unknown"


def load_episode_rows(eval_dir):
    path = f"{eval_dir}/episodes.csv"
    with open(path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row['lap_index'] = int(row['lap_index'])
        row['success'] = row['success'] == 'True'
        row['crashed'] = row['crashed'] == 'True'
        for key in ('lap_time', 'final_progress', 'episode_return', 'final_x', 'final_y', 'fresh_observation_ratio'):
            row[key] = float(row[key])
        if 'mean_fps' in row:
            row['mean_fps'] = float(row['mean_fps'])
        if 'sense_interval' in row:
            row['sense_interval'] = int(row['sense_interval'])
        if 'effective_sensing_hz' in row:
            row['effective_sensing_hz'] = float(row['effective_sensing_hz'])

    return sorted(rows, key=lambda row: row['lap_index'])


def load_step_rows(eval_dir, lap_index):
    path = f"{eval_dir}/lap_{lap_index:02d}/steps.csv"
    with open(path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row['step'] = int(row['step'])
        row['frame_consumed'] = row['frame_consumed'] == 'True'
        for key in ('lap_time', 'x', 'y', 'yaw', 'velocity', 'steering_state',
                    'commanded_steering', 'commanded_speed', 'inst_nav_reward',
                    'cumulative_nav_reward', 'progress'):
            row[key] = float(row[key])
        if 'current_fps' in row:
            row['current_fps'] = float(row['current_fps'])

    return rows


def find_trajectory_npy(eval_dir, lap_index):
    lap_dir = f"{eval_dir}/lap_{lap_index:02d}"
    # AdaptiveFPS's own fixed filename, then fall back to FixedFPS's
    # VehicleStateHistory naming (Lap_<i>_history_<run_name>_<map>.npy)
    for pattern in (f"{lap_dir}/trajectory_lap_{lap_index}.npy", f"{lap_dir}/Lap_*_history_*.npy"):
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No trajectory .npy found under {lap_dir}")


def load_trajectory_npy(path):
    data = np.load(path)
    n_cols = data.shape[1]
    if n_cols == 9:
        cols = NPY_COLS_9
    elif n_cols == 8:
        cols = NPY_COLS_8
    else:
        raise ValueError(f"{path}: expected 8 (AdaptiveFPS) or 9 (FixedFPS) columns, got {n_cols}")

    return {name: data[:, idx] for name, idx in cols.items()}


def load_map_image(map_name):
    """The actual map raster + its world-frame extent, in the same convention
    laser_models.py's ScanSimulator2D.set_map() uses for LiDAR raycasting:
    resolution/origin come from maps/<map>.yaml, and the raw (unflipped) PNG's
    row 0 is the top of the map = the highest y - which is exactly what
    matplotlib's imshow(..., origin='upper') expects."""
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


def plot_single_trajectory(npy_path, map_name, run_label=None):
    lap_time = find_lap_time_for_npy(npy_path)
    trajectory = load_trajectory_npy(npy_path)
    has_fps = "current_fps" in trajectory

    n_panels = 3 if has_fps else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    ax_track, ax_vel = axes[0], axes[-1]

    draw_track(ax_track, map_name)
    scatter = ax_track.scatter(trajectory['x'], trajectory['y'], c=trajectory['velocity'],
                                cmap='viridis', s=8, zorder=1)
    fig.colorbar(scatter, ax=ax_track, label='velocity [m/s]')
    ax_track.set_title(f"Trajectory (velocity): {os.path.basename(npy_path)}")

    if has_fps:
        ax_fps = axes[1]
        draw_track(ax_fps, map_name)

        fps_values = trajectory['current_fps']
        unexpected = sorted(set(np.unique(fps_values)) - set(FPS_CHOICES))
        if unexpected:
            raise ValueError(f"{npy_path}: current_fps contains values outside the valid "
                              f"discrete choices {FPS_CHOICES}: {unexpected}")

        # bucket each sample into its FPS's index (0..3), then use a colormap
        # truncated to exactly 4 levels + integer-width bins - no gradient
        # between the 4 categories, no implied values like 6/7/8/9 Hz.
        fps_index = np.searchsorted(FPS_CHOICES, fps_values)
        cmap = matplotlib.colormaps['viridis'].resampled(len(FPS_CHOICES))
        norm = BoundaryNorm(np.arange(len(FPS_CHOICES) + 1) - 0.5, cmap.N)

        fps_scatter = ax_fps.scatter(trajectory['x'], trajectory['y'], c=fps_index,
                                      cmap=cmap, norm=norm, s=8, zorder=1)
        cbar = fig.colorbar(fps_scatter, ax=ax_fps, ticks=range(len(FPS_CHOICES)))
        cbar.ax.set_yticklabels([f"{fps} Hz" for fps in FPS_CHOICES])
        cbar.set_label('sensing FPS')
        ax_fps.set_title("Trajectory (sensing FPS)")

    ax_vel.plot(trajectory['velocity'])
    ax_vel.set_xlabel('control step')
    ax_vel.set_ylabel('velocity [m/s]')
    ax_vel.set_title('Velocity profile')

    run_part = f"{run_label}   |   " if run_label else ""
    time_label = f"time = {lap_time:.2f}s" if lap_time is not None else "time = unknown"
    fig.suptitle(f"{run_part}{time_label}")
    fig.tight_layout()

    return fig


def plot_overlay(eval_dir, map_name, lap_indices=None):
    episode_rows = load_episode_rows(eval_dir)
    if lap_indices is not None:
        episode_rows = [row for row in episode_rows if row['lap_index'] in lap_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_track(ax, map_name)

    success_labelled, failure_labelled = False, False
    for row in episode_rows:
        steps = load_step_rows(eval_dir, row['lap_index'])
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

        ax.annotate(f"{row['lap_time']:.1f}s", (xs[-1], ys[-1]), fontsize=6,
                    color='0.2', xytext=(3, 3), textcoords='offset points')

    ax.legend()
    ax.set_title(f"{map_name} - {describe_run(episode_rows)} ({eval_dir})\n"
                 f"{sum(r['success'] for r in episode_rows)}/{len(episode_rows)} laps successful")
    fig.tight_layout()

    return fig


def plot_timeseries(eval_dir, lap_indices):
    episode_rows = load_episode_rows(eval_dir)
    lap_times = {row['lap_index']: row['lap_time'] for row in episode_rows}

    per_lap_steps = {lap_index: load_step_rows(eval_dir, lap_index) for lap_index in lap_indices}
    has_fps = any('current_fps' in steps[0] for steps in per_lap_steps.values() if steps)
    if not has_fps:
        print("Note: no current_fps column in these steps.csv rows (FixedFPS output has none) - skipping FPS subplot.")

    n_rows = 4 if has_fps else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows), sharex=True)
    ax_progress, ax_reward, ax_return = axes[0], axes[1], axes[2]
    ax_fps = axes[3] if has_fps else None

    for lap_index in lap_indices:
        steps = per_lap_steps[lap_index]
        control_steps = [step['control_step'] for step in steps]
        label = f"lap {lap_index} ({lap_times[lap_index]:.1f}s)" if lap_index in lap_times else f"lap {lap_index}"

        ax_progress.plot(control_steps, [step['progress'] for step in steps], label=label)
        ax_reward.plot(control_steps, [step['instantaneous_reward'] for step in steps], label=label)
        ax_return.plot(control_steps, [step['cumulative_reward'] for step in steps], label=label)
        if has_fps:
            ax_fps.plot(control_steps, [step.get('current_fps') for step in steps], label=label)

    ax_progress.set_ylabel('progress')
    ax_reward.set_ylabel('instantaneous reward')
    ax_return.set_ylabel('cumulative reward')
    if has_fps:
        ax_fps.set_ylabel('sensing FPS [Hz]')
    axes[-1].set_xlabel('control step')
    ax_progress.legend()
    ax_progress.set_title(f"{describe_run(episode_rows)} ({eval_dir})")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True, help="lap-data directory, e.g. AdaptiveFPS/eval/f1_aut/fixed_5Hz or FixedFPS/eval/f1_aut/interval_2")
    parser.add_argument("--map", default=DEFAULT_MAP_NAME, help="map name (for drawing the track)")
    parser.add_argument("--laps", type=int, nargs='+', default=None, help="restrict to these lap indices")
    parser.add_argument("--npy-lap", type=int, default=None, help="plot one lap's trajectory .npy (track colored by velocity, and by FPS if available)")
    parser.add_argument("--overlay", action='store_true', help="overlay successful/failed trajectories")
    parser.add_argument("--timeseries", action='store_true', help="plot progress/reward/return(/FPS) over time for --laps")
    parser.add_argument("--output-dir", default="AdaptiveFPS/analysis", help="directory where plots are saved")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    eval_dir = args.eval_dir.rstrip('/')
    label = run_label_for(eval_dir)

    if not (args.npy_lap is not None or args.overlay or args.timeseries):
        parser.error("nothing to do: pass --npy-lap, --overlay, or --timeseries")

    if args.npy_lap is not None:
        npy_path = find_trajectory_npy(eval_dir, args.npy_lap)
        episode_rows = load_episode_rows(eval_dir)
        run_desc = describe_run(episode_rows)
        fig = plot_single_trajectory(npy_path, args.map, run_label=run_desc)
        save_figure(fig, args.output_dir, args.map, label, "trajectory", f"lap{args.npy_lap}")

    if args.overlay:
        fig = plot_overlay(eval_dir, args.map, lap_indices=args.laps)
        extra = "laps" + "-".join(str(lap) for lap in args.laps) if args.laps else None
        save_figure(fig, args.output_dir, args.map, label, "overlay", extra)

    if args.timeseries:
        laps = args.laps
        if laps is None:
            laps = [row['lap_index'] for row in load_episode_rows(eval_dir)]
        fig = plot_timeseries(eval_dir, laps)
        extra = "laps" + "-".join(str(lap) for lap in laps)
        save_figure(fig, args.output_dir, args.map, label, "timeseries", extra)


if __name__ == '__main__':
    main()
