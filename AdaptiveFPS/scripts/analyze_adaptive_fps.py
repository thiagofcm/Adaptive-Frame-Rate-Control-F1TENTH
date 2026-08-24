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

Figures are saved as paper-ready PNG+PDF pairs under:

    AdaptiveFPS/paper_media/trajectories/<track>/fc_<frame_cost>/

Sensing budget is not part of that hierarchy (multiple budgets share the
same track/frame-cost folder) - it appears in filenames and figure titles
instead.

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
import re
from pathlib import Path

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
DEFAULT_PAPER_MEDIA_ROOT = "AdaptiveFPS/paper_media"

# Maximum number of overlaid laps for which per-lap endpoint time labels
# stay readable; beyond this they all cluster near the finish line and
# become illegible clutter, so they're dropped in favor of trajectory
# structure and success/failure behavior.
MAX_ANNOTATED_ENDPOINTS = 8

# Shared publication-style rcParams, matching the other AdaptiveFPS paper
# figures (plot_sensing_trade_off.py / plot_sensing_trade_off_bars.py /
# plot_frame_acquisition.py).
PAPER_RCPARAMS = {
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 1.1,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

# Two possible trajectory .npy layouts, told apart by column count:
#  - 9 cols: FixedFPS/TestSimulation_FixedFPS_Refactor.py's VehicleStateHistory
#    format - full_states (RaceCar.state: x,y,steering_angle,velocity,yaw,
#    yaw_rate,slip_angle) + 2 action columns.
#  - 8 cols: AdaptiveFPS/scripts/evaluate_adaptive_fps.py's own format -
#    AdaptiveFPSEnv never exposes full_states, so this uses the TAL 5-element
#    state (x,y,yaw,velocity,steering) + 2 action columns + current_fps.
NPY_COLS_9 = {"x": 0, "y": 1, "steering_state": 2, "velocity": 3, "yaw": 4, "commanded_steering": 7, "commanded_speed": 8}
NPY_COLS_8 = {"x": 0, "y": 1, "yaw": 2, "velocity": 3, "steering_state": 4, "commanded_steering": 5, "commanded_speed": 6, "current_fps": 7}


def parse_run_label(run_label):
    """Extract (frame_cost, budget) from an adaptive run directory name such
    as 'adaptive_fc_0.075_bud_300.0' or 'adaptive_fc_0.075_bud_300.0_bp_10.0'.
    Returns (None, None) for anything else (e.g. 'fixed_5Hz', 'interval_2') -
    that's the normal, expected case for FixedFPS runs, not an error."""
    match = re.search(r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)", str(run_label))
    if match is None:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def format_fc_title(frame_cost):
    """Concise frame-cost value for titles/legends, e.g. 0.075 -> '0.075',
    0.0 -> '0' (no unnecessary trailing zeros)."""
    return "0" if frame_cost == 0.0 else f"{frame_cost:g}"


def format_fc_folder(frame_cost):
    """Deterministic frame-cost folder suffix. Unlike format_fc_title, 0.0
    stays '0.0' (not '0') so the folder name is unambiguous on disk."""
    return "0.0" if frame_cost == 0.0 else f"{frame_cost:g}"


def format_track_title(track):
    """'f1_aut' -> 'F1 AUT' - generic, works for any track name."""
    return track.upper().replace("_", " ")


def format_condition(frame_cost, budget, fallback_desc):
    """Concise experimental-condition string for titles: '$f_c=..., B=...$'
    for adaptive runs, or the FixedFPS-style description otherwise."""
    if frame_cost is not None and budget is not None:
        return f"$f_c={format_fc_title(frame_cost)}$, $B={int(budget)}$"
    return fallback_desc


def get_paper_output_dir(paper_media_root, track, frame_cost=None, run_label=None):
    """AdaptiveFPS/paper_media/trajectories/<track>/fc_<frame_cost>/ for
    adaptive runs. Budget is deliberately not part of this hierarchy (it
    goes in filenames/titles instead). For non-adaptive runs (no frame
    cost/budget parsed, e.g. FixedFPS), fall back to the run's own label
    as the subfolder rather than crashing."""
    if frame_cost is not None:
        subfolder = f"fc_{format_fc_folder(frame_cost)}"
    else:
        subfolder = run_label or "unknown_run"

    out_dir = Path(paper_media_root) / "trajectories" / track / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_filename(plot_type, budget=None, extra=None):
    """Budget isn't part of the folder hierarchy, so it goes in the
    filename instead to keep same-folder runs from overwriting each other,
    e.g. 'trajectory_b300_lap03', 'overlay_b150'."""
    parts = [plot_type]
    if budget is not None:
        parts.append(f"b{int(budget)}")
    if extra:
        parts.append(extra)
    return "_".join(parts)


def save_figure(fig, out_dir, filename_stem):
    """Save a paper-ready PNG (300 DPI) + vector PDF pair, close only once
    both are written, and report both paths."""
    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return png_path, pdf_path


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
    # alpha < 1 keeps the map as visually secondary context so the
    # trajectory itself dominates the panel.
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=extent, origin='upper', alpha=0.85, zorder=0)

    track = StdTrack(map_name)
    ax.plot(track.wpts[:, 0], track.wpts[:, 1], '--', color='tab:orange', linewidth=1,
            zorder=1, label='Track Centerline')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def find_lap_time_for_npy(npy_path):
    """Look up this lap's total time from the steps.csv saved alongside the
    .npy (same lap_<i>/ directory) - the .npy itself has no time column."""
    steps_path = os.path.join(os.path.dirname(npy_path), "steps.csv")
    if not os.path.exists(steps_path):
        return None

    with open(steps_path, 'r', newline='') as file:
        rows = list(csv.DictReader(file))

    return float(rows[-1]['lap_time']) if rows else None


def plot_single_trajectory(npy_path, map_name, lap_index, condition):
    lap_time = find_lap_time_for_npy(npy_path)
    trajectory = load_trajectory_npy(npy_path)
    has_fps = "current_fps" in trajectory

    n_panels = 3 if has_fps else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    ax_track, ax_vel = axes[0], axes[-1]

    draw_track(ax_track, map_name)
    scatter = ax_track.scatter(trajectory['x'], trajectory['y'], c=trajectory['velocity'],
                                cmap='viridis', s=10, zorder=2)
    fig.colorbar(scatter, ax=ax_track, label='Velocity (m/s)')
    ax_track.set_title("Trajectory — Velocity")

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
                                      cmap=cmap, norm=norm, s=10, zorder=2)
        cbar = fig.colorbar(fps_scatter, ax=ax_fps, ticks=range(len(FPS_CHOICES)))
        cbar.ax.set_yticklabels([f"{fps} Hz" for fps in FPS_CHOICES])
        cbar.set_label('Sensing Rate (Hz)')
        ax_fps.set_title("Trajectory — Sensing Rate")

    ax_vel.plot(trajectory['velocity'], color='tab:blue', linewidth=1.3)
    ax_vel.set_xlabel('Control Step')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Velocity Profile')
    ax_vel.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.35)
    ax_vel.set_axisbelow(True)
    for spine in ax_vel.spines.values():
        spine.set_linewidth(1.1)

    time_label = f"{lap_time:.1f} s" if lap_time is not None else "time unknown"
    fig.suptitle(f"Lap {lap_index} — {condition} — {time_label}")
    fig.tight_layout()

    return fig


def plot_overlay(eval_dir, map_name, track_title, condition, lap_indices=None):
    episode_rows = load_episode_rows(eval_dir)
    if lap_indices is not None:
        episode_rows = [row for row in episode_rows if row['lap_index'] in lap_indices]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_track(ax, map_name)

    annotate_endpoints = len(episode_rows) <= MAX_ANNOTATED_ENDPOINTS
    if not annotate_endpoints:
        print(f"Note: {len(episode_rows)} laps overlaid - skipping per-lap endpoint time "
              f"labels (would overlap); showing trajectory structure only.")

    success_labelled, failure_labelled, failure_loc_labelled = False, False, False
    for row in episode_rows:
        steps = load_step_rows(eval_dir, row['lap_index'])
        xs = [step['x'] for step in steps]
        ys = [step['y'] for step in steps]

        if row['success']:
            ax.plot(xs, ys, color='tab:green', alpha=0.6, linewidth=1,
                     label='Successful' if not success_labelled else None, zorder=2)
            success_labelled = True
        else:
            ax.plot(xs, ys, color='tab:red', alpha=0.6, linewidth=1,
                     label='Failed' if not failure_labelled else None, zorder=2)
            ax.plot(row['final_x'], row['final_y'], 'x', color='black', markersize=8, zorder=3,
                    label='Failure Location' if not failure_loc_labelled else None)
            failure_labelled = True
            failure_loc_labelled = True

        if annotate_endpoints:
            ax.annotate(f"{row['lap_time']:.1f}s", (xs[-1], ys[-1]), fontsize=6,
                        color='0.2', xytext=(3, 3), textcoords='offset points')

    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black',
              borderpad=0.6, handletextpad=0.6)

    n_success = sum(row['success'] for row in episode_rows)
    success_pct = 100.0 * n_success / len(episode_rows) if episode_rows else float('nan')
    ax.set_title(f"Evaluation Trajectories — {track_title}\n"
                 f"{condition} — Success Rate: {success_pct:.0f}%")
    fig.tight_layout()

    return fig


def plot_timeseries(eval_dir, lap_indices, track_title, condition):
    episode_rows = load_episode_rows(eval_dir)
    lap_times = {row['lap_index']: row['lap_time'] for row in episode_rows}

    per_lap_steps = {lap_index: load_step_rows(eval_dir, lap_index) for lap_index in lap_indices}
    has_fps = any('current_fps' in steps[0] for steps in per_lap_steps.values() if steps)
    if not has_fps:
        print("Note: no current_fps column in these steps.csv rows (FixedFPS output has none) - skipping FPS subplot.")

    n_rows = 4 if has_fps else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.3 * n_rows), sharex=True)
    ax_progress, ax_reward, ax_return = axes[0], axes[1], axes[2]
    ax_fps = axes[3] if has_fps else None

    show_legend = len(lap_indices) <= MAX_ANNOTATED_ENDPOINTS
    if not show_legend:
        print(f"Note: {len(lap_indices)} laps requested - omitting the per-lap legend to avoid clutter.")

    for lap_index in lap_indices:
        steps = per_lap_steps[lap_index]
        control_steps = [step['step'] for step in steps]
        label = f"Lap {lap_index} ({lap_times[lap_index]:.1f}s)" if lap_index in lap_times else f"Lap {lap_index}"

        ax_progress.plot(control_steps, [step['progress'] for step in steps], label=label, linewidth=1.3)
        ax_reward.plot(control_steps, [step['inst_nav_reward'] for step in steps], label=label, linewidth=1.3)
        ax_return.plot(control_steps, [step['cumulative_nav_reward'] for step in steps], label=label, linewidth=1.3)
        if has_fps:
            ax_fps.plot(control_steps, [step.get('current_fps') for step in steps], label=label, linewidth=1.3)

    ax_progress.set_ylabel('Track Progress')
    ax_reward.set_ylabel('Instantaneous Reward')
    ax_return.set_ylabel('Cumulative Reward')
    if has_fps:
        ax_fps.set_ylabel('Sensing Rate (Hz)')
    axes[-1].set_xlabel('Control Step')

    for ax in axes:
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    if show_legend:
        ax_progress.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black',
                            borderpad=0.6, handletextpad=0.6)

    fig.suptitle(f"Episode Dynamics — {track_title}\n{condition}")
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
    parser.add_argument("--output-dir", default=DEFAULT_PAPER_MEDIA_ROOT, help="paper_media root directory (figures are organized under trajectories/<track>/fc_<frame_cost>/ beneath it)")
    args = parser.parse_args()

    plt.rcParams.update(PAPER_RCPARAMS)

    eval_dir = args.eval_dir.rstrip('/')
    label = run_label_for(eval_dir)
    track_title = format_track_title(args.map)

    if not (args.npy_lap is not None or args.overlay or args.timeseries):
        parser.error("nothing to do: pass --npy-lap, --overlay, or --timeseries")

    frame_cost, budget = parse_run_label(label)
    out_dir = get_paper_output_dir(args.output_dir, args.map, frame_cost, run_label=label)

    if args.npy_lap is not None:
        npy_path = find_trajectory_npy(eval_dir, args.npy_lap)
        episode_rows = load_episode_rows(eval_dir)
        condition = format_condition(frame_cost, budget, describe_run(episode_rows))
        fig = plot_single_trajectory(npy_path, args.map, args.npy_lap, condition)
        filename = build_filename("trajectory", budget, f"lap{args.npy_lap:02d}")
        save_figure(fig, out_dir, filename)

    if args.overlay:
        episode_rows_for_condition = load_episode_rows(eval_dir)
        condition = format_condition(frame_cost, budget, describe_run(episode_rows_for_condition))
        fig = plot_overlay(eval_dir, args.map, track_title, condition, lap_indices=args.laps)
        extra = "laps" + "-".join(f"{lap:02d}" for lap in args.laps) if args.laps else None
        filename = build_filename("overlay", budget, extra)
        save_figure(fig, out_dir, filename)

    if args.timeseries:
        laps = args.laps
        if laps is None:
            laps = [row['lap_index'] for row in load_episode_rows(eval_dir)]
        condition = format_condition(frame_cost, budget, describe_run(load_episode_rows(eval_dir)))
        fig = plot_timeseries(eval_dir, laps, track_title, condition)
        extra = "laps" + "-".join(f"{lap:02d}" for lap in laps)
        filename = build_filename("timeseries", budget, extra)
        save_figure(fig, out_dir, filename)


if __name__ == '__main__':
    main()
