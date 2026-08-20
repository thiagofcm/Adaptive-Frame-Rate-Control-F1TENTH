"""
AdaptiveFPSEnv - Gymnasium environment for learning the adaptive LiDAR
sensing-frequency policy, with the frozen TAL navigation actor unchanged
underneath it.

Sibling to FixedFPS/ (the validated fixed-interval reference), not a
dependent of it - this file imports only the original TAL components,
never anything from FixedFPS/.

The adaptive action controls only sensing frequency (self.fps_choices);
steering/speed still come entirely from the frozen AgentTester/FastArchitecture
pipeline. Validated in fixed-FPS mode via AdaptiveFPS/scripts/evaluate_adaptive_fps.py
(Stage 3B). Registered as Gymnasium id "AdaptiveFPS-v0" for
AdaptiveFPS/scripts/train_adaptive_fps_ppo.py's CleanRL PPO+LSTM loop (Stage 4).
"""
import numpy as np
import torch
import gymnasium
from gymnasium import spaces

from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import setup_run_list, load_conf
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack
from TrajectoryAidedLearning.Utils.RewardSignals import TALearningReward
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester

# physics substep duration, mirrors the hardcoded default in f110_env.py /
# base_classes.py (F110Env/RaceCar are constructed without a timestep kwarg)
PHYSICS_TIMESTEP = 0.01


class AdaptiveFPSEnv(gymnasium.Env):
    def __init__(self, run_file, map_name, model_run_name, budget, frame_cost, budget_penalty=10.0):
        super().__init__()

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

        assert self.run.architecture == "fast", "AdaptiveFPSEnv only supports the frozen TAL ('fast') planner"
        self.planner = AgentTester(self.run, self.conf)

        # progress-only track - deliberately never wired into anything that
        # gates collision detection (see _build_observation)
        self.progress_track = StdTrack(self.run.map_name)
        self.reward = TALearningReward(self.conf, self.run) # same reward class training used

        # ---------------------------------------------------------------
        # Control-time abstraction: one step() = one _physics_step() =
        # conf.sim_steps physics substeps held under one navigation action.
        # Individual substeps are never exposed to the caller.
        # ---------------------------------------------------------------
        self.control_frequency = 1.0 / (PHYSICS_TIMESTEP * self.conf.sim_steps) # = 10.0 Hz

        # ---------------------------------------------------------------
        # FPS action space. [1, 2, 5, 10] control-step intervals are never
        # hardcoded anywhere - always computed inline from fps_choices and
        # control_frequency, at the point of use (reset() and step()).
        # ---------------------------------------------------------------
        self.fps_choices = [10, 5, 2, 1]
        for fps in self.fps_choices:
            assert (self.control_frequency / fps).is_integer(), \
                f"{fps} Hz does not evenly divide {self.control_frequency} Hz control rate"
        self.action_space = spaces.Discrete(len(self.fps_choices))

        # ---------------------------------------------------------------
        # Gymnasium/PPO-facing observation: the single currently-held LiDAR
        # scan (self.last_sampled_scan), normalized with TAL's own scan
        # scaling (same formula FastArchitecture.transform_obs uses: divide
        # by range_finder_scale, clip to [0, 1]) but WITHOUT TAL's n_scans=2
        # stacking buffer - PPO's LSTM handles temporal history itself -
        # plus 3 real temporal/adaptive-sensing variables (fps_ratio,
        # obs_age_ratio, frame_ratio - see _get_augmented_obs). Distinct
        # from observation_for_navigation (built in step() below), which
        # still feeds AgentTester/FastArchitecture completely unchanged.
        # ---------------------------------------------------------------
        # budget: nominal max frames-consumed-per-episode, only used (for
        # now) to normalize frame_ratio in the PPO observation - not wired
        # into the reward yet.
        self.budget = budget
        # frame_cost: placeholder for a later reward-shaping stage - unused here.
        self.frame_cost = frame_cost
        # budget_penalty: flat reward override applied once per episode, the
        # first tick episode_frame_count exceeds budget (see step()).
        self.budget_penalty = budget_penalty
        # max_obs_interval: control steps between fresh reads at the lowest
        # available FPS - the normalization denominator for obs_age_ratio.
        self.max_obs_interval = int(self.control_frequency / min(self.fps_choices))

        self.temporal_state_dim = 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.conf.n_beams + self.temporal_state_dim,), dtype=np.float32)

        # episode state, (re)initialized in reset()
        self.current_observation = None
        self.last_sampled_scan = None
        self.steps_since_last_obs = 0
        self.episode_frame_count = 0
        self.current_fps = None
        self.obs_interval = None
        self.prev_obs = None
        self.prev_action = None
        self.budget_penalty_applied = False
        self.cumulative_reward = 0.0
        self.progress = 0.0

    def _get_augmented_obs(self, obs_values):
        """PPO-facing observation only - never fed to AgentTester/FastArchitecture.
        Same range_finder_scale normalization TAL uses, single scan (no
        n_scans stacking), plus 3 real temporal/adaptive-sensing variables:

          fps_ratio:     self.current_fps / self.control_frequency
                          (10/5/2/1 Hz -> 1.0/0.5/0.2/0.1)
          obs_age_ratio: self.steps_since_last_obs / self.max_obs_interval,
                          clipped to [0, 1] - how old the currently held
                          LiDAR observation is, in control-step units.
          frame_ratio:   self.episode_frame_count / self.budget, clipped to
                          [0, 1] - how much of the episode's nominal frame
                          budget has been consumed so far.

        None of these three feed the reward yet.
        """
        normalized_scan = np.clip(obs_values / self.conf.range_finder_scale, 0.0, 1.0)

        fps_ratio = self.current_fps / self.control_frequency
        obs_age_ratio = np.clip(self.steps_since_last_obs / self.max_obs_interval, 0.0, 1.0)
        episode_frame_count = np.clip(self.episode_frame_count / self.budget, 0.0, 1.0)
        temporal_state = np.array([fps_ratio, obs_age_ratio, episode_frame_count], dtype=np.float32)

        return np.concatenate([normalized_scan, temporal_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        reset_pose = np.zeros(3)[None, :]
        obs, step_reward, done, _ = self.env.reset(reset_pose)

        self.world_step_count = 0
        self.prev_obs = None
        self.prev_action = None
        observation = self._build_observation(obs, done)
        self.current_observation = observation
        

        # Seed the held scan from the real initial scan, so the first
        # step()'s navigation call has a valid scan to use.
        self.last_sampled_scan = self.current_observation["scan"].copy()
        self.steps_since_last_obs = 0

        # The reset scan does NOT count as a consumed frame: episode_frame_count
        # only increments inside step()'s sensing-gate branch, i.e. it counts
        # frames acquired as a result of an adaptive decision. The reset scan
        # is the deterministic initial condition, not a policy choice.
        self.episode_frame_count = 0
        self.budget_penalty_applied = False

        # Initial FPS = 10 Hz -> obs_interval = 1, so the very first step()
        # call is guaranteed to hit the sensing gate (steps_since_last_obs(1)
        # >= obs_interval(1)) - step()'s very first action always gets used
        # to pick the next interval, no ambiguous "no decision yet" state.
        self.current_fps = 10
        self.obs_interval = int(self.control_frequency / self.current_fps)

        self.cumulative_reward = 0.0
        self.progress = 0.0

        ppo_observation = self._get_augmented_obs(self.last_sampled_scan)

        # DEBUG PRINT
        # print(f"RESET STEP")
        # print("----------------------------------------------------------------------------------------------------------------")

        return ppo_observation, {}

    def step(self, action):
        self.world_step_count +=1
        self.steps_since_last_obs += 1

        # ---------------------------------
        # 1. Frozen TAL navigation uses currently held LiDAR
        # ---------------------------------
        observation_for_navigation = dict(self.current_observation)
        observation_for_navigation["scan"] = self.last_sampled_scan

        self.prev_obs = self.current_observation
        navigation_action = self.planner.plan(observation_for_navigation)

        # ---------------------------------
        # 2. One control transition
        # ---------------------------------
        observation, nav_reward, terminated, truncated, phys_info = self._physics_step(navigation_action)
        self.current_observation = observation

        # ---------------------------------
        # 3. Sampling timer
        # ---------------------------------
        if self.steps_since_last_obs >= self.obs_interval:
            self.last_sampled_scan = self.current_observation["scan"].copy()

            self.steps_since_last_obs = 0
            self.episode_frame_count += 1

            frame_consumed = True
            lidar_fresh = True

            # Action selected at this sampling instant controls the FUTURE
            # sensing rate - never retroactive to the navigation call above,
            # which already used the previously-held scan.
            self.current_fps = self.fps_choices[int(action)]
            self.obs_interval = int(self.control_frequency / self.current_fps)
        else:
            frame_consumed = False
            lidar_fresh = False

        self.progress = max(self.progress, phys_info["progress"])
        frame_penalty = self.frame_cost if frame_consumed else 0.0
        reward = nav_reward - frame_penalty

        if not observation["lap_done"] and self.episode_frame_count > self.budget and not self.budget_penalty_applied:
            reward = -self.budget_penalty
            self.budget_penalty_applied = True

        self.cumulative_reward += reward

        # ---------------------------------
        # 4. Adaptive-policy observation (PPO-facing only)
        # ---------------------------------
        ppo_observation = self._get_augmented_obs(self.last_sampled_scan)

        # DEBUG PRINT
        # print(
        # f"Step {self.world_step_count}: "
        # f"nav_action={navigation_action}, "
        # f"nav_reward={nav_reward:.3f}, "
        # f"frame_penalty={frame_penalty:.3f}, "
        # f"adaptive_fps_reward={reward:.3f}, "
        # f"progress={phys_info['progress']:.3f}, "
        # f"fps={self.current_fps}, "
        # f"obs_interval={self.obs_interval}, "
        # f"steps_since_last_obs={self.steps_since_last_obs}, "
        # f"lidar_fresh={lidar_fresh}, "
        # f"frame_consumed={frame_consumed}, "
        # f"episode_frame_count={self.episode_frame_count}, "
        # f"cumulative_reward={self.cumulative_reward:.3f}, "
        # f"fps_ratio={ppo_observation[-3]:.3f}, "
        # f"obs_age_ratio={ppo_observation[-2]:.3f}, "
        # f"episode_frame_count_ratio={ppo_observation[-1]:.3f}"
        # )
        #print("----------------------------------------------------------------------------------------------------------------")

        # ---------------------------------
        # 5. Debug info
        # ---------------------------------
        info = {
            "navigation_action": navigation_action,
            "nav_reward": nav_reward,
            "reward": reward,
            "current_fps": self.current_fps,
            "obs_interval": self.obs_interval,
            "steps_since_last_obs": self.steps_since_last_obs,
            "lidar_fresh": lidar_fresh,
            "frame_consumed": frame_consumed,
            "episode_frame_count": self.episode_frame_count,
            "progress": self.progress,
            "cumulative_reward": self.cumulative_reward,
            "frame_penalty": frame_penalty,
            "budget_penalty_applied": self.budget_penalty_applied
        }

        if terminated:
            self.planner.lap_complete()

        return ppo_observation, reward, terminated, truncated, info

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


gymnasium.register(id="AdaptiveFPS-v0", entry_point="AdaptiveFPS.envs.adaptive_fps_env:AdaptiveFPSEnv")
