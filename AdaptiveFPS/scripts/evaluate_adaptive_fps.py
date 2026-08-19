"""
Evaluation script for AdaptiveFPSEnv - fixed-FPS mode for now (Stage 3B),
structured so an adaptive PPO mode can be dropped in later without
restructuring the loop.

The explicit branch below is intentional and NOT simplified to
`action = FPS_TO_ACTION[fixed_fps]`, even though --fixed is required today:
later the else branch becomes `action = adaptive_policy(...)`.

AdaptiveFPSEnv itself is untouched here - this script only ever calls
env.reset()/env.step(action) and reads its public info dict / current_observation.

Output is FPS-centric (mean_fps), not interval-centric - AdaptiveFPS's own
CSVs never mention sense_interval. This is intentionally a different schema
from FixedFPS/eval/.../episodes.csv (interval-centric); compare by value
using the Hz<->interval mapping (10<->1, 5<->2, 2<->5, 1<->10), not by column
name.

Run from the repo root:
    python -m AdaptiveFPS.scripts.evaluate_adaptive_fps --fixed 5 --n-laps 100
"""
import argparse
import atexit
import csv
import glob
import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import numpy as np

from AdaptiveFPS.envs.adaptive_fps_env import AdaptiveFPSEnv

FPS_TO_ACTION = {10: 0, 5: 1, 2: 2, 1: 3}
LSTM_HIDDEN_SIZE = 64

MODEL_RUN_NAME_DEFAULT = "fast_Std_Std_TAL_f1_aut_6_5_0"
MAP_NAME_DEFAULT = "f1_aut"
RUN_FILE = "TAL_maps"

EVAL_ROOT = "AdaptiveFPS/eval"
EPISODE_CSV_FIELDS = [
    "lap_index", "success", "crashed", "mean_fps", "lap_time", "final_progress",
    "episode_return", "nav_episode_return", "steps_length", "n_fresh_observations",
    "fresh_observation_ratio", "final_x", "final_y",
]
STEP_CSV_FIELDS = [
    "step", "lap_time", "x", "y", "yaw", "velocity", "steering_state",
    "commanded_steering", "commanded_speed", "frame_consumed",
    "inst_nav_reward",
    "inst_frame_penalty",
    "inst_adaptive_reward",
    "cumulative_nav_reward",
    "cumulative_adaptive_reward",
    "progress", "current_fps",
]
# [x, y, yaw, velocity, steering_state, commanded_steering, commanded_speed, current_fps] -
# necessarily a different layout from VehicleStateHistory's 9-column full_states-based
# one: AdaptiveFPSEnv._build_observation() never exposes full_states/yaw_rate/slip_angle.
TRAJECTORY_NPY_COLUMNS = 8

SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_CSV_FIELDS = [
    "run_name", "mean_fps", "n_laps", "success_rate", "mean_lap_time", "std_lap_time",
    "mean_episode_return", "std_episode_return", "mean_final_progress",
    "mean_n_fresh_observations", "std_n_fresh_observations",
]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentEval(nn.Module):
    def __init__(self, obs_dim, n_actions, lstm_hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, lstm_hidden_size)

        # Same LSTM initialization used during training
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape(
            (-1, batch_size, self.lstm.input_size)
        )
        done = done.reshape((-1, batch_size))
        new_hidden = []

        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(
            torch.cat(new_hidden),0,1)

        return new_hidden, lstm_state

    def predict(self,obs,lstm_state,done,deterministic=True,):

        obs_tensor = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor(
            [float(done)],
            dtype=torch.float32
        )

        with torch.no_grad():

            hidden, lstm_state = self.get_states(obs_tensor,lstm_state,done_tensor)
            logits = self.actor(hidden)
            if deterministic:
                action = torch.argmax(logits,dim=-1)
            else:
                action = Categorical(logits=logits).sample()

        return int(action.item()), lstm_state

def summarize_results(map_name):
    """Read every */episodes.csv under AdaptiveFPS/eval/<map_name>/ - both
    fixed_*Hz/ (fixed-FPS baselines) and adaptive_fc_*_bud_*/ (trained-model
    evaluations) - and combine them into one comparison table: one row per
    run (run_name = the directory containing that run's episodes.csv), with
    mean_fps and mean/std lap_time/episode_return/final_progress/
    n_fresh_observations/success_rate all computed across every evaluated
    lap for that run.

    Registered to run on interpreter exit so it always reflects the latest
    accumulated results, whether the run finished normally or was interrupted.
    """
    map_dir = f"{EVAL_ROOT}/{map_name}"
    # generic (not fixed_*Hz-only) so any evaluated-config subdirectory with
    # its own episodes.csv is picked up, regardless of naming convention -
    # previously this only globbed fixed_*Hz/, silently excluding
    # adaptive_fc_*_bud_*/ runs from the summary.
    episode_csv_paths = sorted(glob.glob(f"{map_dir}/*/episodes.csv"))
    if not episode_csv_paths:
        return

    summary_rows = []
    for path in episode_csv_paths:
        with open(path, 'r', newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue

        run_name = os.path.basename(os.path.dirname(path))

        lap_times = np.array([float(row['lap_time']) for row in rows])
        successes = np.array([row['success'] == 'True' for row in rows]) # each lap's success as a bool
        returns = np.array([float(row['episode_return']) for row in rows])
        final_progress = np.array([float(row['final_progress']) for row in rows])
        n_fresh_observations = np.array([float(row['n_fresh_observations']) for row in rows])
        # mean of every episode's own mean_fps, not just rows[0]'s - fixed-FPS
        # runs have an identical value every lap, but an adaptive policy's
        # mean_fps varies lap to lap, so row[0] alone would silently be wrong.
        mean_fps_per_episode = np.array([float(row['mean_fps']) for row in rows])

        summary_rows.append({
            "run_name": run_name,
            "mean_fps": float(mean_fps_per_episode.mean()),
            "n_laps": len(rows),
            "success_rate": float(successes.mean() * 100),
            "mean_lap_time": float(lap_times.mean()),
            "std_lap_time": float(lap_times.std()),
            "mean_episode_return": float(returns.mean()),
            "std_episode_return": float(returns.std()),
            "mean_final_progress": float(final_progress.mean()),
            "mean_n_fresh_observations": float(n_fresh_observations.mean()),
            "std_n_fresh_observations": float(n_fresh_observations.std()),
        })

    summary_rows.sort(key=lambda row: row['mean_fps'])

    summary_path = f"{map_dir}/{SUMMARY_CSV_NAME}"
    with open(summary_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("_________________________________________________________")
    print(f"Run summary for map '{map_name}' (from all */episodes.csv found on disk):")
    for row in summary_rows:
        print(f"  {row['run_name']:<28} mean_fps={row['mean_fps']:>5.2f}  n_laps={row['n_laps']:>3}  "
              f"success_rate={row['success_rate']:6.2f}%  "
              f"lap_time mean={row['mean_lap_time']:.4f} std={row['std_lap_time']:.4f}  "
              f"return mean={row['mean_episode_return']:.4f} std={row['std_episode_return']:.4f}  "
              f"fresh_obs mean={row['mean_n_fresh_observations']:.2f} std={row['std_n_fresh_observations']:.2f}")
    print(f"Summary written to: {summary_path}")


def run_lap(env, fixed_fps, lap_dir, model=None, lap_index=None):

    adaptive_episode_return = 0.0
    nav_episode_return = 0.0
    observation, info = env.reset()

    lstm_state = (
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
    )
    done = False

    control_step_count = 0
    n_fresh_observations = 0
    fps_trace = []
    step_rows = []
    trajectory_rows = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        control_step_index = control_step_count
        pre_step_state = env.current_observation["state"]

        if fixed_fps is not None:
            action = FPS_TO_ACTION[fixed_fps]
        else:
            action, lstm_state = model.predict(
                observation,
                lstm_state,
                done,
                deterministic=True,
            )
            # raise NotImplementedError("Adaptive PPO evaluation is not implemented yet.")

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        control_step_count += 1

        adaptive_episode_return += reward
        nav_episode_return += info["nav_reward"]

        if info["frame_consumed"]:
            n_fresh_observations += 1

        fps_trace.append(info["current_fps"])

        step_rows.append({
            "step": control_step_index,
            "lap_time": env.current_observation["current_laptime"],
            "x": pre_step_state[0],
            "y": pre_step_state[1],
            "yaw": pre_step_state[2],
            "velocity": pre_step_state[3],
            "steering_state": pre_step_state[4],
            "commanded_steering": info["navigation_action"][0],
            "commanded_speed": info["navigation_action"][1],
            "frame_consumed": info["frame_consumed"],
            "inst_nav_reward": info["nav_reward"],
            "cumulative_nav_reward": nav_episode_return,
            "inst_adaptive_reward": reward,
            "cumulative_adaptive_reward": adaptive_episode_return,
            "inst_frame_penalty": info["frame_penalty"],
            "progress": info["progress"],
            "current_fps": info["current_fps"],
        })
        trajectory_rows.append([
            pre_step_state[0], pre_step_state[1], pre_step_state[2],
            pre_step_state[3], pre_step_state[4],
            info["navigation_action"][0], info["navigation_action"][1],
            info["current_fps"],
        ])

    true_obs = env.current_observation
    assert n_fresh_observations == info["episode_frame_count"], \
        f"fresh-observation count mismatch: counted {n_fresh_observations}, env reports {info['episode_frame_count']}"

    # closing row: final state, dummy [0, 0] action, but retain the last real
    # current_fps so the trajectory can be colored by FPS end-to-end with no
    # spurious 0 at the last point.
    trajectory_rows.append([
        true_obs["state"][0], true_obs["state"][1], true_obs["state"][2],
        true_obs["state"][3], true_obs["state"][4],
        0.0, 0.0, info["current_fps"],
    ])

    os.makedirs(lap_dir, exist_ok=True)
    with open(f"{lap_dir}/steps.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=STEP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    np.save(f"{lap_dir}/trajectory_lap_{lap_index}.npy", np.array(trajectory_rows, dtype=np.float64))

    return {
        "mean_fps": float(np.mean(fps_trace)) if fps_trace else 0.0,
        "success": true_obs["lap_done"],
        "crashed": true_obs["colision_done"],
        "lap_time": true_obs["current_laptime"],
        "final_progress": info["progress"],
        "episode_return": adaptive_episode_return,
        "nav_episode_return": nav_episode_return,
        "steps_length": control_step_count,
        "n_fresh_observations": n_fresh_observations,
        "fresh_observation_ratio": n_fresh_observations / control_step_count if control_step_count else 0.0,
        "final_x": true_obs["state"][0],
        "final_y": true_obs["state"][1],
        # console-only diagnostics, not written to episodes.csv:
        "final_current_fps": info["current_fps"],
        "final_obs_interval": info["obs_interval"],
        "episode_frame_count": info["episode_frame_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=int, default=None, choices=[10, 5, 2, 1],
                         help="constant sensing FPS to drive AdaptiveFPSEnv with every control step")
    parser.add_argument("--n-laps", type=int, default=100)
    parser.add_argument("--nav-model", default=MODEL_RUN_NAME_DEFAULT)
    parser.add_argument("--model", help="PPO model run name to load from AdaptiveFPS/models/<run_name>/best_model.pt")
    parser.add_argument("--fc", type=float, default=0.0, required=True,
                        help="frame cost to use for adaptive PPO evaluation (ignored if --fixed is set)")
    parser.add_argument("--bud", type=float, default=300.0, required=True,
                        help="budget to use for adaptive PPO evaluation (ignored if --fixed is set)")
    parser.add_argument("--map", default=MAP_NAME_DEFAULT)
    args = parser.parse_args()

    fixed_fps = args.fixed
    navigation_model = args.nav_model
    map_name = args.map
    budget = args.bud
    frame_cost = args.fc

    atexit.register(summarize_results, map_name)

    env = AdaptiveFPSEnv(RUN_FILE, map_name, navigation_model,budget,frame_cost)

    if fixed_fps is not None:
        out_dir = f"{EVAL_ROOT}/{map_name}/fixed_{fixed_fps}Hz"
        model = None
        model_label = f"fixed_{fixed_fps}Hz"
    else:
        checkpoint = torch.load(args.model,map_location="cpu")
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n

        model = AgentEval(obs_dim=obs_dim,n_actions=n_actions,lstm_hidden_size=LSTM_HIDDEN_SIZE)

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        print(f"Loaded PPO model: {args.model}")
        model_label = f"adaptive_fc_{args.fc}"
        out_dir = f"{EVAL_ROOT}/{map_name}/adaptive_fc_{args.fc}_bud_{args.bud}"

    episode_rows = []
    for lap_index in range(args.n_laps):
        lap_dir = f"{out_dir}/lap_{lap_index:02d}"
        row = run_lap(env, fixed_fps, lap_dir, model, lap_index)
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


if __name__ == '__main__':
    main()
