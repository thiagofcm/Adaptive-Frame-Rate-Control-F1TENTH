"""
FixedSensingF110Env - reusable single-episode wrapper around F110Env + the
frozen TAL navigation actor.

This class deliberately does NOT own any sensing-freshness concept: no
sense_interval, no modulo logic, no last_fresh_scan cache, no freshness
counters. control_step() takes whatever LiDAR scan the caller decides is
currently available and runs physics + frozen TAL navigation with it. The
temporal sensing gate lives in the caller (see
FixedFPS/scripts/evaluate_fixed_sensing_env.py) - that seam is where a
future adaptive-FPS policy gets plugged in, so it must stay visible there,
not buried inside this class.

Stage 2 of the adaptive-FPS refactor path: NOT the PPO/adaptive env yet.
Public API is reset()/control_step(available_scan) only - no step(action),
no Gymnasium. See TrajectoryAidedLearning/TestSimulation_FixedSampling.py
(untouched reference) and TestSimulation_FixedFPS_Refactor.py (untouched,
validated code-motion refactor) for the validated behavior this class
reproduces.
"""
import numpy as np
import torch

from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import setup_run_list, load_conf
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack
from TrajectoryAidedLearning.Utils.RewardSignals import TALearningReward
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester


class FixedSensingF110Env:
    def __init__(self, run_file, map_name, model_run_name):
        run_data = setup_run_list(run_file)
        matches = [r for r in run_data if r.map_name == map_name and r.run_name == model_run_name]
        assert matches, f"No run found for map={map_name} run_name={model_run_name} in {run_file}"
        self.run = matches[0]
        self.conf = load_conf("config_file")

        seed = self.run.random_seed + 10 * self.run.n
        np.random.seed(seed) # repetition seed
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)

        self.noise_rng = None
        self.noise_std = 0.0
        if self.run.noise_std > 0:
            self.noise_std = self.run.noise_std
            self.noise_rng = np.random.default_rng(seed=seed)

        self.env = F110Env(map=self.run.map_name)
        self.map_name = self.run.map_name

        assert self.run.architecture == "fast", "FixedSensingF110Env only supports the frozen TAL ('fast') planner"
        self.planner = AgentTester(self.run, self.conf)

        # progress-only track - deliberately never wired into anything that
        # gates collision detection (see _build_observation)
        self.progress_track = StdTrack(self.run.map_name)
        self.reward = TALearningReward(self.conf, self.run) # same reward class training used

        # episode state, (re)initialized in reset() - NOTE: no sense_interval,
        # no last_fresh_scan, no freshness counters. The env doesn't know
        # what "fresh" means - that's the caller's job.
        self.current_observation = None
        self.prev_obs = None
        self.prev_action = None
        self.cumulative_reward = 0.0
        self.progress = 0.0

    def reset(self):
        self.cumulative_reward = 0.0
        self.progress = 0.0

        reset_pose = np.zeros(3)[None, :]
        obs, step_reward, done, _ = self.env.reset(reset_pose)

        self.prev_obs = None
        self.prev_action = None
        observation = self._build_observation(obs, done)
        self.current_observation = observation

        return observation, {}

    def control_step(self, available_scan):
        """available_scan: whatever LiDAR scan the caller has decided is
        currently available to navigation (fresh or a stale cached one) -
        the env does not decide this."""

        # A. Build navigation observation from the TRUE current state, with
        #    only the scan swapped for whatever the caller supplied.
        observation_for_navigation = dict(self.current_observation)
        observation_for_navigation["scan"] = available_scan

        # B. Frozen TAL navigation.
        self.prev_obs = self.current_observation
        navigation_action = self.planner.plan(observation_for_navigation)

        # C. Physical transition.
        observation, nav_reward, terminated, truncated, info = self._physics_step(navigation_action)
        self.current_observation = observation

        self.progress = max(self.progress, info["progress"])
        self.cumulative_reward += nav_reward

        info.update({
            "progress": self.progress,
            "cumulative_reward": self.cumulative_reward,
            "navigation_action": navigation_action,
        })
        # deliberately no "lidar_fresh"/"n_fresh_observations" here - the env
        # doesn't own the sensing gate, so it has no freshness state to report.

        if terminated:
            self.planner.lap_complete()

        return observation, nav_reward, terminated, truncated, info

    def _physics_step(self, navigation_action):
        self.prev_action = navigation_action

        sim_steps = self.conf.sim_steps
        done = False
        while sim_steps > 0 and not done:
            obs, step_reward, done, raw_info = self.env.step(navigation_action[None, :])
            sim_steps -= 1

        observation = self._build_observation(obs, done)

        progress = self.progress_track.calculate_progress_percent(observation["state"][0:2])
        nav_reward = observation["reward"]
        terminated = observation["lap_done"] or observation["colision_done"]
        truncated = False
        info = {"progress": progress}

        return observation, nav_reward, terminated, truncated, info

    def _build_observation(self, obs, done):
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
        observation['scan'] = obs['scans'][0]

        if self.noise_rng:
            noise = self.noise_rng.normal(scale=self.noise_std, size=2)
        else:
            noise = np.zeros(2)
        pose_x = obs['poses_x'][0] + noise[0]
        pose_y = obs['poses_y'][0] + noise[1]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        if done and obs['lap_counts'][0] == 0:
            observation['colision_done'] = True
        if obs['lap_counts'][0] == 1:
            observation['lap_done'] = True

        observation['reward'] = self.reward(observation, self.prev_obs, self.prev_action)

        return observation

    def close(self):
        self.env.close_rendering()
