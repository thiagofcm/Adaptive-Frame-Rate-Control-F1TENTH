# Run from the repo root, e.g.:
#     python -m AdaptiveFPS.scripts.train_adaptive_fps_ppo --total-timesteps 20000
import os
os.environ["OMP_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]   = "1"
os.environ["OPENBLAS_NTHREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# AdaptiveFPSEnv.__init__ calls torch.use_deterministic_algorithms(True) (a
# global flag, inherited unmodified from the original TAL seeding pattern -
# not something this stage touches). CuBLAS ops on CUDA can't run
# deterministically without this set first, or the Agent's forward pass on
# GPU raises RuntimeError. Must be set before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import random
import time
import sys
import yaml
import argparse
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit
from datetime import datetime

import AdaptiveFPS.envs.adaptive_fps_env  # noqa: F401 - side effect: registers "AdaptiveFPS-v0"

RUNS_ROOT = "AdaptiveFPS/runs"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "AdaptiveFPS-v0"
    """the id of the environment"""
    run_file: str = "TAL_maps"
    """TAL run-config file (config/<run_file>.yaml) AdaptiveFPSEnv loads the run/conf from"""
    map_name: str = "f1_aut"
    """map name"""
    model_run_name: str = "fast_Std_Std_TAL_f1_aut_6_5_0"
    """frozen TAL navigation model run_name"""
    total_timesteps: int = 20_000
    """total timesteps of the experiments (small default - integration test, not a real training run)"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (small default: each env owns a full F110Env
    + its own copy of the frozen TAL actor, heavier per-instance than LunarLander)"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.98
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    max_episode_steps: int = 300
    """TimeLimit wrapper cutoff - comfortably above the known ~220-step successful-lap length"""
    resume_path: str = None
    """path to a checkpoint .pt file to resume training from"""
    async_envs: bool = False
    """use AsyncVectorEnv (one real subprocess per env) instead of SyncVectorEnv. Defaults to
    False here (unlike the LunarLander reference, which defaults True): F110Env's fork-safety
    under multiprocessing (numba-jitted raycasting, any matplotlib rendering state) hasn't been
    verified yet, so this first integration run keeps stepping sequential. Try True once Sync
    round-trips cleanly."""

    budget: int = 300
    """nominal frame budget per episode"""
    frame_cost: float = 0.0
    """penalty applied when a fresh frame is consumed"""
    budget_penalty: float = 10.0
    """flat reward override applied once per episode, the first tick episode_frame_count exceeds budget"""

def make_env(env_id, run_file, map_name, model_run_name, max_episode_steps, budget, frame_cost, budget_penalty):
    def thunk():
        env = gym.make(env_id, run_file=run_file, map_name=map_name, model_run_name=model_run_name, budget=budget, frame_cost=frame_cost, budget_penalty=budget_penalty)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, lstm_hidden_size=64):
        super().__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, lstm_hidden_size)
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
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        """
        Run input network + LSTM.
        Resets hidden state automatically when done=True (episode boundary).

        x:          (n_envs, obs_dim)
        lstm_state: ((1, batch(n_envs), hidden), (1, batch(n_envs), hidden))
        done:       (batch,)
        """
        hidden = self.network(x)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
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

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None, deterministic=False):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


def load_args(args_class):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, remaining = pre.parse_known_args()

    base = args_class()
    if known.config:
        with open(known.config) as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if not hasattr(base, k):
                raise KeyError(f"Unknown key in {known.config}: '{k}'")
            setattr(base, k, v)

    sys.argv = [sys.argv[0]] + remaining
    return tyro.cli(args_class, default=base)


if __name__ == "__main__":
    args = load_args(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    run_name = f"{args.env_id}_{args.map_name}_{date_str}_fc_{args.frame_cost}_bud_{args.budget}_bp_{args.budget_penalty}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{RUNS_ROOT}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    info_file = os.path.join(f"{RUNS_ROOT}/{run_name}", "info_settings.txt")
    os.makedirs(f"{RUNS_ROOT}/{run_name}", exist_ok=True)
    with open(info_file, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        if args.resume_path is not None:
            f.write(f"Resumed from checkpoint: {args.resume_path}\n")
    print(f"Experiment info saved -> {info_file}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_fns = [make_env(args.env_id, args.run_file, args.map_name, args.model_run_name, args.max_episode_steps, args.budget, args.frame_cost, args.budget_penalty)
               for _ in range(args.num_envs)]
    if args.async_envs:
        envs = gym.vector.AsyncVectorEnv(env_fns, context="fork")
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_iteration = checkpoint["iteration"] + 1
        next_lstm_state = (
            checkpoint["next_lstm_state_h"].to(device),
            checkpoint["next_lstm_state_c"].to(device),
        )
        print(f"Resumed from checkpoint: {args.resume_path} (iteration {checkpoint['iteration']}, step {global_step})")
    else:
        global_step = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # 1.0 where the fps_action was actually applied this tick (a real sampling
    # instant, AdaptiveFPSEnv's info["frame_consumed"]), 0.0 where it was
    # silently discarded (obs_interval not yet elapsed) -- used to exclude
    # non-causal ticks from the policy loss (see update loop below). Defaults
    # to 0 so an unset slot doesn't accidentally get treated as a real decision.
    masks = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Chosen FPS / reward-decomposition diagnostics, accumulated per-env across an episode
    episode_fps_sum = np.zeros(args.num_envs)
    episode_fps_count = np.zeros(args.num_envs)
    episode_nav_reward_sum = np.zeros(args.num_envs)
    episode_nav_reward_count = np.zeros(args.num_envs)
    episode_frame_penalty_sum = np.zeros(args.num_envs)

    for iteration in range(start_iteration, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # -----------------------------------------------------------------
        # ROLLOUT COLLECTION.
        # -----------------------------------------------------------------
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # frame_consumed: was the fps_action this tick a real decision
            # (obs_interval elapsed) or silently discarded? Guard against the
            # vector-env slot where step() didn't actually run this tick (an
            # env that just auto-reset) the same way as chosen_fps below.
            if "frame_consumed" in infos:
                valid_fc = infos.get("_frame_consumed", np.ones(args.num_envs, dtype=bool))
                fc = np.where(valid_fc, infos["frame_consumed"], 0.0)
                masks[step] = torch.tensor(fc, dtype=torch.float32).to(device)

            # log chosen fps / nav_reward every step -- skip envs that just
            # auto-reset this step: SyncVectorEnv/AsyncVectorEnv reset on the
            # *next* step() call after termination, and reset()'s info dict
            # doesn't carry "current_fps"/"nav_reward", so infos["_current_fps"][i]
            # (etc.) is False for those envs.
            if "current_fps" in infos:
                valid_fps = infos.get("_current_fps", np.ones(args.num_envs, dtype=bool))
                for i in range(args.num_envs):
                    if valid_fps[i]:
                        episode_fps_sum[i] += infos["current_fps"][i]
                        episode_fps_count[i] += 1
            if "nav_reward" in infos:
                valid_nav = infos.get("_nav_reward", np.ones(args.num_envs, dtype=bool))
                for i in range(args.num_envs):
                    if valid_nav[i]:
                        episode_nav_reward_sum[i] += infos["nav_reward"][i]
                        episode_nav_reward_count[i] += 1
            if "frame_penalty" in infos:
                valid_frame_penalty = infos.get("_frame_penalty", np.ones(args.num_envs, dtype=bool))
                for i in range(args.num_envs):
                    if valid_frame_penalty[i]:
                        # signed contribution to reward (reward = nav_reward - frame_penalty),
                        # so episodic_nav_reward + episodic_frame_penalty reconstructs episodic_return
                        episode_frame_penalty_sum[i] += -infos["frame_penalty"][i]

            if "episode" in infos:
                finished = infos["episode"]["_r"]
                for i, done in enumerate(finished):
                    if done:
                        ep_return = infos["episode"]["r"][i]
                        ep_length = infos["episode"]["l"][i]

                        mean_fps = episode_fps_sum[i] / episode_fps_count[i] if episode_fps_count[i] > 0 else 0
                        mean_nav_reward = episode_nav_reward_sum[i] / episode_nav_reward_count[i] if episode_nav_reward_count[i] > 0 else 0
                        episodic_nav_reward = episode_nav_reward_sum[i]
                        episodic_frame_penalty = episode_frame_penalty_sum[i]
                        episode_frame_count = infos["episode_frame_count"][i] if "episode_frame_count" in infos else 0

                        print(f"global_step={global_step} | return={ep_return:.2f} | steps={ep_length} | mean_fps={mean_fps:.2f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        writer.add_scalar("charts/mean_chosen_fps", mean_fps, global_step)
                        writer.add_scalar("charts/mean_nav_reward", mean_nav_reward, global_step)
                        writer.add_scalar("charts/episodic_nav_reward", episodic_nav_reward, global_step)
                        writer.add_scalar("charts/episodic_frame_penalty", episodic_frame_penalty, global_step)
                        writer.add_scalar("charts/episode_frame_count", episode_frame_count, global_step)

                        episode_fps_sum[i] = 0
                        episode_fps_count[i] = 0
                        episode_nav_reward_sum[i] = 0
                        episode_nav_reward_count[i] = 0
                        episode_frame_penalty_sum[i] = 0

        # -----------------------------------------------------------------
        # COMPUTING ADVANTAGES AND bootstrap value if EP in env not done
        # -----------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # -----------------------------------------------------------------
        # PREPARING FOR UPDATE
        # -----------------------------------------------------------------
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)
        b_masks = masks.reshape(-1)

        assert args.num_envs % args.num_minibatches == 0
        envs_per_minibatch = args.num_envs // args.num_minibatches
        envs_indices = np.arange(args.num_envs)
        timestep_env_grid = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envs_indices)
            for start in range(0, args.num_envs, envs_per_minibatch):
                end = start + envs_per_minibatch
                minibatch_env_indices = envs_indices[start:end]
                minibatch_flat_indices = timestep_env_grid[:, minibatch_env_indices].ravel()
                minibatch_dones = dones[:, minibatch_env_indices].reshape(-1)

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[minibatch_flat_indices],
                    (
                        initial_lstm_state[0][:, minibatch_env_indices],
                        initial_lstm_state[1][:, minibatch_env_indices],
                    ),
                    minibatch_dones,
                    b_actions.long()[minibatch_flat_indices],
                )
                logratio = newlogprob - b_logprobs[minibatch_flat_indices]
                ratio = logratio.exp()

                # mb_mask: 1.0 where fps_action was a real, causal decision
                # this tick, 0.0 where it was silently discarded. Policy-side
                # terms (pg_loss, entropy, KL/clipfrac diagnostics) must not
                # train on the discarded-action ticks -- the sampled action
                # there had zero effect on the transition that produced the
                # reward being credited, so training on it is wrong credit
                # assignment, not just extra variance. v_loss/GAE targets are
                # unaffected -- "how good is this state" is meaningful every
                # tick regardless of whether that tick's action was causal.
                mb_mask = b_masks[minibatch_flat_indices]
                mb_mask_sum = mb_mask.sum().clamp(min=1.0)

                with torch.no_grad():
                    old_approx_kl = (mb_mask * (-logratio)).sum() / mb_mask_sum
                    approx_kl = (mb_mask * ((ratio - 1) - logratio)).sum() / mb_mask_sum
                    clipfracs += [((mb_mask * ((ratio - 1.0).abs() > args.clip_coef).float()).sum() / mb_mask_sum).item()]

                mb_advantages = b_advantages[minibatch_flat_indices]
                if args.norm_adv:
                    # Normalize using only the causal-tick statistics --
                    # otherwise the discarded-tick advantages (the majority,
                    # at low FPS) would dominate the mean/std used to rescale
                    # the minority that actually feeds the policy loss below.
                    valid_advantages = mb_advantages[mb_mask.bool()]
                    if valid_advantages.numel() > 0:
                        mb_advantages = (mb_advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = (mb_mask * torch.max(pg_loss1, pg_loss2)).sum() / mb_mask_sum

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[minibatch_flat_indices]) ** 2
                    v_clipped = b_values[minibatch_flat_indices] + torch.clamp(
                        newvalue - b_values[minibatch_flat_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch_flat_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch_flat_indices]) ** 2).mean()

                entropy_loss = (mb_mask * entropy).sum() / mb_mask_sum
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # -- Checkpoint saving --
        if iteration % 10 == 0:
            checkpoint_path = f"{RUNS_ROOT}/{run_name}/ckpts/timestep_{global_step}_iterations_{iteration}"
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_model_path = f"{checkpoint_path}/ckpt_{global_step}_iterations_{iteration}.pt"
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "global_step": global_step,
                "iteration": iteration,
                "next_lstm_state_h": next_lstm_state[0].cpu(),
                "next_lstm_state_c": next_lstm_state[1].cpu(),
            }, checkpoint_model_path)
            print(f"  Checkpoint saved -> {checkpoint_model_path}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"iteration={iteration}/{args.num_iterations} global_step={global_step} SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "global_step": global_step,
    }, f"{RUNS_ROOT}/{run_name}/model.pt")
    print(f"Model saved -> {RUNS_ROOT}/{run_name}/model.pt")
    writer.close()
