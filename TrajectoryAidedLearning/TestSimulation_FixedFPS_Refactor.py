from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack
from TrajectoryAidedLearning.Utils.RewardSignals import TALearningReward

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester

import torch
import numpy as np
import time
import csv
import os
import glob
import atexit

# physics substep duration, mirrors the hardcoded default in f110_env.py /
# base_classes.py (F110Env/RaceCar are constructed without a timestep kwarg)
PHYSICS_TIMESTEP = 0.01

# settings
SHOW_TRAIN = False
# SHOW_TEST = False
SHOW_TEST = False
VERBOSE = True
LOGGING = True

# sample-and-hold LiDAR settings
SENSE_INTERVAL = 5
N_TEST_LAPS = 20
MODEL_RUN_NAME = "fast_Std_Std_TAL_f1_aut_6_5_0"
MAP_NAME = "f1_aut"

# Everything this experiment writes lives under one root, one subfolder per
# sense_interval, one subfolder per lap within that - so sweeping SENSE_INTERVAL
# across repeated runs never overwrites another interval's (or lap's) results.
# Tagged "FixedFPS_Refactor" (not "FixedSampling") so this refactored script's
# output never overwrites TestSimulation_FixedSampling.py's validated reference
# results - the two can be diffed directly to regression-test this refactor.
#   Data/FixedSamplingResults/<model>/FixedFPS_Refactor/interval_<N>/episodes.csv
#   Data/FixedSamplingResults/<model>/FixedFPS_Refactor/interval_<N>/lap_<i>/steps.csv
#   Data/FixedSamplingResults/<model>/FixedFPS_Refactor/interval_<N>/lap_<i>/Lap_<i>_history_..._<map>.npy
FIXED_SAMPLING_ROOT = f"Data/FixedSamplingResults/{MODEL_RUN_NAME}/FixedFPS_Refactor"

EPISODE_CSV_NAME = "episodes.csv"
EPISODE_CSV_FIELDS = [
    "sense_interval", "effective_sensing_hz", "lap_index", "success", "crashed",
    "lap_time", "final_progress", "episode_return", "n_control_steps",
    "n_fresh_observations", "fresh_observation_ratio", "final_x", "final_y",
]

STEP_CSV_NAME = "steps.csv"
STEP_CSV_FIELDS = [
    "control_step", "lap_time", "x", "y", "yaw", "velocity", "steering_state",
    "commanded_steering", "commanded_speed", "lidar_fresh",
    "instantaneous_reward", "cumulative_reward", "progress",
]

# regenerated on exit from every interval_*/episodes.csv found under FIXED_SAMPLING_ROOT
SUMMARY_CSV_PATH = f"Data/FixedSamplingResults/{MODEL_RUN_NAME}_fixedfps_refactor_summary.csv"
SUMMARY_CSV_FIELDS = [
    "sense_interval", "n_laps", "success_rate", "mean_lap_time", "std_lap_time",
    "mean_episode_return", "std_episode_return", "mean_final_progress",
]


def summarize_results():
    """Read every interval_*/episodes.csv under FIXED_SAMPLING_ROOT and summarize
    mean/std lap_time, episode_return, final_progress and success_rate per
    sense_interval, across however many laps are recorded for it.

    Registered to run on interpreter exit so it always reflects the latest
    accumulated results, whether the run finished normally or was interrupted.
    """
    episode_csv_paths = sorted(glob.glob(f"{FIXED_SAMPLING_ROOT}/interval_*/{EPISODE_CSV_NAME}"))
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

    with open(SUMMARY_CSV_PATH, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("_________________________________________________________")
    print("Sense-interval summary (from all interval_*/episodes.csv found on disk):")
    for row in summary_rows:
        print(f"  interval={row['sense_interval']:>3}  n_laps={row['n_laps']:>3}  "
              f"success_rate={row['success_rate']:6.2f}%  "
              f"lap_time mean={row['mean_lap_time']:.4f}  std={row['std_lap_time']:.4f}  "
              f"return mean={row['mean_episode_return']:.4f}")
    print(f"Summary written to: {SUMMARY_CSV_PATH}")

class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None

        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.prev_obs = None
        self.prev_action = None

        self.std_track = None
        self.map_name = None
        self.reward = None
        self.noise_rng = None

        # flags
        self.vehicle_state_history = None

        # sample-and-hold LiDAR state
        self.sense_interval = SENSE_INTERVAL
        self.last_fresh_scan = None
        self.last_step_was_fresh = True
        self.control_step = 0
        self.n_fresh_scans = 0

        # logging/analysis-only state - never read by the navigation pipeline.
        # progress_track is intentionally separate from self.std_track: wiring
        # it into self.std_track would additionally enable StdTrack.check_done()
        # inside build_observation(), which can mark laps as crashed on
        # backward progress - a behavior change this experiment must not make.
        self.progress_track = None
        self.interval_dir = None
        self.lap_progress = 0.0
        self.cumulative_reward = 0.0
        self.step_rows = []
        self.episode_rows = []

    def run_testing_evaluation(self):
        for run in self.run_data:
            print(run)
            print("_________________________________________________________")
            print(run.run_name)
            print("_________________________________________________________")
            seed = run.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

            if run.noise_std > 0:
                self.noise_std = run.noise_std
                self.noise_rng = np.random.default_rng(seed=seed)

            self.env = F110Env(map=run.map_name)
            self.map_name = run.map_name

            if run.architecture == "PP":
                planner = PurePursuit(self.conf, run)
            elif run.architecture == "fast":
                planner = AgentTester(run, self.conf)
            else: raise AssertionError(f"Planner {run.planner} not found")

            if run.test_mode == "Std": self.planner = planner
            else: raise AssertionError(f"Test mode {run.test_mode} not found")

            self.vehicle_state_history = VehicleStateHistory(run, "Testing/")

            # logging-only additions: neither read by AgentTester/FastArchitecture
            # nor by anything that decides actions, collisions, or lap timing.
            self.progress_track = StdTrack(run.map_name)
            self.reward = TALearningReward(self.conf, run) # same reward class training used

            self.interval_dir = f"{FIXED_SAMPLING_ROOT}/interval_{self.sense_interval}"
            os.makedirs(self.interval_dir, exist_ok=True)
            self.episode_rows = []

            self.n_test_laps = run.n_test_laps
            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()

            self.env.close_rendering()

    def run_testing(self):
        assert self.env != None, "No environment created"
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:

                control_step_index = self.control_step
                pre_step_state = observation['state']
                self.prev_obs = observation

                # ---------------------------------
                # 1. Fixed sensing / temporal gate
                # ---------------------------------

                self.last_step_was_fresh = (self.control_step % self.sense_interval == 0)

                if self.last_step_was_fresh:
                    self.last_fresh_scan = observation['scan']
                    self.n_fresh_scans += 1

                self.control_step += 1

                observation_for_navigation = dict(observation)
                observation_for_navigation['scan'] = self.last_fresh_scan

                lidar_fresh = self.last_step_was_fresh

                # ---------------------------------
                # 2. Frozen TAL navigation
                # ---------------------------------

                navigation_action = self.planner.plan(observation_for_navigation)

                # ---------------------------------
                # 3. Physical/control transition
                # ---------------------------------

                observation, nav_reward, terminated, truncated, info = self._physics_step(navigation_action)

                # existing reward/progress/logging follows unchanged:
                self.lap_progress = max(self.lap_progress, info["progress"])
                self.cumulative_reward += nav_reward

                self.step_rows.append({
                    "control_step": control_step_index,
                    "lap_time": observation['current_laptime'],
                    "x": pre_step_state[0],
                    "y": pre_step_state[1],
                    "yaw": pre_step_state[2],
                    "velocity": pre_step_state[3],
                    "steering_state": pre_step_state[4],
                    "commanded_steering": navigation_action[0],
                    "commanded_speed": navigation_action[1],
                    "lidar_fresh": lidar_fresh,
                    "instantaneous_reward": nav_reward,
                    "cumulative_reward": self.cumulative_reward,
                    "progress": self.lap_progress,
                })

            self.planner.lap_complete()
            if observation['lap_done']:
                if VERBOSE: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if VERBOSE: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")

            effective_sensing_hz = (1.0 / (PHYSICS_TIMESTEP * self.conf.sim_steps)) / self.sense_interval
            self.episode_rows.append({
                "sense_interval": self.sense_interval,
                "effective_sensing_hz": effective_sensing_hz,
                "lap_index": i,
                "success": observation['lap_done'],
                "crashed": observation['colision_done'],
                "lap_time": observation['current_laptime'],
                "final_progress": self.lap_progress,
                "episode_return": self.cumulative_reward,
                "n_control_steps": self.control_step,
                "n_fresh_observations": self.n_fresh_scans,
                "fresh_observation_ratio": self.n_fresh_scans / self.control_step if self.control_step else 0.0,
                "final_x": observation['state'][0],
                "final_y": observation['state'][1],
            })

            lap_dir = f"{self.interval_dir}/lap_{i:02d}"
            os.makedirs(lap_dir, exist_ok=True)
            self.write_step_csv(lap_dir)

            # VehicleStateHistory itself is untouched; only its output `.path`
            # is redirected here so different intervals/laps stop overwriting
            # each other's trajectory .npy under the original Testing/ folder.
            if self.vehicle_state_history:
                self.vehicle_state_history.path = lap_dir + "/"
                self.vehicle_state_history.save_history(i, test_map=self.map_name)

        self.write_episode_csv()

        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)

        return eval_dict

    def _physics_step(self, navigation_action):
        """One navigation/control step: apply `navigation_action` to F110Env for
        conf.sim_steps physics substeps (self.run_step, unchanged), rebuild the
        observation (self.build_observation, unchanged), and surface this step's
        progress/reward/termination - identical values to what run_testing()'s
        loop already computed inline, just returned instead of mutating instance
        state directly.
        """
        observation = self.run_step(navigation_action)
        if SHOW_TEST: self.env.render('human_fast')

        progress = self.progress_track.calculate_progress_percent(observation['state'][0:2])
        nav_reward = observation['reward']
        terminated = observation['lap_done'] or observation['colision_done']
        truncated = False
        info = {"progress": progress}

        return observation, nav_reward, terminated, truncated, info

    # this is an overide
    def run_step(self, action):
        sim_steps = self.conf.sim_steps
        if self.vehicle_state_history:
            self.vehicle_state_history.add_action(action)
        self.prev_action = action

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1

        observation = self.build_observation(obs, done)

        return observation

    def build_observation(self, obs, done):
        """Build observation

        Returns
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams

        """
        observation = {}
        observation['current_laptime'] = obs['lap_times'][0]
        observation['scan'] = obs['scans'][0] #TODO: introduce slicing here

        if self.noise_rng:
            noise = self.noise_rng.normal(scale=self.noise_std, size=2)
        else: noise = np.zeros(2)
        pose_x = obs['poses_x'][0] + noise[0]
        pose_y = obs['poses_y'][0] + noise[1]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        observation['reward'] = 0.0
        if done and obs['lap_counts'][0] == 0:
            observation['colision_done'] = True
        if self.std_track is not None:
            if self.std_track.check_done(observation) and obs['lap_counts'][0] == 0:
                observation['colision_done'] = True

            if self.prev_obs is None: observation['progress'] = 0
            elif self.prev_obs['lap_done'] == True: observation['progress'] = 0
            else: observation['progress'] = max(self.std_track.calculate_progress_percent(state[0:2]), self.prev_obs['progress'])
            # self.racing_race_track.plot_vehicle(state[0:2], state[2])
            # taking the max progress


        if obs['lap_counts'][0] == 1:
            observation['lap_done'] = True

        if self.reward:
            observation['reward'] = self.reward(observation, self.prev_obs, self.prev_action)

        if self.vehicle_state_history:
            self.vehicle_state_history.add_state(obs['full_states'][0])

        return observation

    def write_episode_csv(self):
        """One row per lap for this sense_interval - overwritten each run of
        this interval, since it always covers the full self.episode_rows."""
        path = f"{self.interval_dir}/{EPISODE_CSV_NAME}"

        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.episode_rows)

        print(f"Episode results written to: {path}")

    def write_step_csv(self, lap_dir):
        """One row per control step for the lap that just finished."""
        path = f"{lap_dir}/{STEP_CSV_NAME}"

        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=STEP_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.step_rows)

    def reset_simulation(self):
        self.control_step = 0
        self.last_fresh_scan = None
        self.n_fresh_scans = 0
        self.lap_progress = 0.0
        self.cumulative_reward = 0.0
        self.step_rows = []

        reset_pose = np.zeros(3)[None, :]

        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if SHOW_TRAIN: self.env.render('human_fast')

        self.prev_obs = None
        observation = self.build_observation(obs, done)
        # self.prev_obs = observation
        if self.std_track is not None:
            self.std_track.max_distance = 0.0

        return observation

def main():
    atexit.register(summarize_results)

    run_file = "TAL_maps"
    sim = TestSimulation(run_file)

    sim.run_data = [
        run for run in sim.run_data
        if run.map_name == MAP_NAME and run.run_name == MODEL_RUN_NAME
    ]
    sim.run_data[0].n_test_laps = N_TEST_LAPS

    sim.run_testing_evaluation()


if __name__ == '__main__':
    main()
