#!/usr/bin/env python3
"""Sweep runner for the AdaptiveFPS frame-cost experiments.

Reads a sweep config (default: AdaptiveFPS/config/fc_sweep.yaml) containing
every base training.py hyperparameter plus a `sweep:` section mapping
parameter names to lists of values. Builds the Cartesian product of the swept
parameters, materializes one full config file per combination, and launches
AdaptiveFPS/scripts/train_adaptive_fps_ppo.py once per combination via
`python -m` (repo-root CWD, matching how every other script in this project
is invoked) -- each pinned to its own isolated CPU core block via taskset.

Does not modify AdaptiveFPSEnv, train_adaptive_fps_ppo.py, PPO/reward logic,
or FixedFPS/*. Only manages experiments (config generation + subprocess
launch/logging) - all training logic stays in train_adaptive_fps_ppo.py.

CPU-only by design (no GPU/CUDA_VISIBLE_DEVICES logic here at all) - the
sweep config's own `cuda: false` is simply respected as-is.

CPU allocation is intentionally simple: `sorted(os.sched_getaffinity(0))[:n]`,
no contention-ranking. On a heavily shared machine, two independent sweep
launches can still collide on the same cores this way (each process gets
CPU-affinity-restricted to that core, not exclusive use of it - they just get
time-sliced ~50/50). Pass --cpus explicitly if you know another sweep might
already be running; a contention-aware picker (ranking cores by how many
other single-affinity processes already claim them, e.g. via psutil) is a
natural upgrade later if this bites in practice.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAIN_MODULE = "AdaptiveFPS.scripts.train_adaptive_fps_ppo"
DEFAULT_SWEEP_CONFIG = os.path.join(REPO_ROOT, "AdaptiveFPS", "config", "fc_sweep.yaml")
SWEEPS_ROOT = os.path.join(REPO_ROOT, "AdaptiveFPS", "sweeps")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=DEFAULT_SWEEP_CONFIG,
                    help="yaml file with base hyperparameters + a 'sweep:' section of param -> list of values")
    p.add_argument("--total-timesteps", type=int, default=None,
                    help="override total_timesteps for every run in the sweep (default: whatever the sweep config has)")
    p.add_argument("--cpus", type=int, nargs="+", default=None,
                    help="flat pool of CPU core ids to partition across runs, via taskset -- 1 core per "
                         "combo normally, or num_envs+1 per combo when that combo's config has "
                         "async_envs=true (see compute_cores_per_combo); omit to auto-pick "
                         "sorted(os.sched_getaffinity(0))[:n_needed]")
    p.add_argument("--max-parallel", type=int, default=5,
                    help="max number of training runs active at once (conservative default - this is a "
                         "shared machine; each combo can claim several cores when async_envs is on)")
    p.add_argument("--extra", type=str, default="",
                    help="extra CLI args forwarded verbatim to train_adaptive_fps_ppo.py")
    return p.parse_args()


def load_sweep_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    sweep_params = cfg.pop("sweep", None)
    if not sweep_params:
        raise ValueError(f"No non-empty 'sweep:' section found in {path}")
    return cfg, sweep_params


def sweep_combinations(sweep_params):
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def combo_tag(overrides):
    return "_".join(f"{k}{v}" for k, v in overrides.items())


def compute_cores_per_combo(base_cfg, overrides):
    """How many CPU cores this combo needs: 1 normally, or num_envs+1 (one core per
    AsyncVectorEnv worker subprocess, +1 for the main process) when async_envs is on.

    Async workers are forked from the main process and inherit its CPU affinity mask
    at fork time -- taskset-ing the whole combo to a single core would silently pin
    every worker to that same one core too, giving Async zero real parallelism.
    """
    merged = {**base_cfg, **overrides}
    if merged.get("async_envs", False):
        return int(merged.get("num_envs", 1)) + 1
    return 1


def detect_cpus(n_needed):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < n_needed:
        raise ValueError(f"Sweep needs {n_needed} CPU cores total, only {len(available)} available "
                          f"via os.sched_getaffinity(0); pass --cpus explicitly to override.")
    return available[:n_needed]


def launch(run_cfg_path, cpu_block, args, log_dir, tag):
    cpu_str = ",".join(str(c) for c in cpu_block)
    cmd = ["taskset", "-c", cpu_str, sys.executable, "-u", "-m", TRAIN_MODULE, "--config", run_cfg_path]
    if args.extra:
        cmd += args.extra.split()

    log_path = os.path.join(log_dir, f"{tag}.log")
    log_file = open(log_path, "w")
    print(f"[sweep] starting {tag} on cpus={cpu_str} -> {log_path}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path


def main():
    args = parse_args()
    base_cfg, sweep_params = load_sweep_config(args.config)
    combos = list(sweep_combinations(sweep_params))

    cores_needed = [compute_cores_per_combo(base_cfg, overrides) for overrides in combos]
    total_cores_needed = sum(cores_needed)

    cpus = args.cpus if args.cpus is not None else detect_cpus(total_cores_needed)
    if len(cpus) < total_cores_needed:
        raise ValueError(f"{len(combos)} combinations need {total_cores_needed} distinct CPU cores total "
                          f"(sum of per-combo requirements), only got {len(cpus)} via --cpus")

    # Partition the flat core pool into one contiguous-in-order block per combo.
    cpu_blocks = []
    idx = 0
    for n in cores_needed:
        cpu_blocks.append(cpus[idx:idx + n])
        idx += n

    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    run_root = os.path.join(SWEEPS_ROOT, f"sweep_{date_str}")
    cfg_dir = os.path.join(run_root, "configs")
    log_dir = os.path.join(run_root, "logs")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # (tag, run_cfg_path, cpu_block) queued in order; popped as slots free up.
    queue = []
    for i, overrides in enumerate(combos):
        merged = {**base_cfg, **overrides}
        if args.total_timesteps is not None:
            merged["total_timesteps"] = args.total_timesteps

        tag = combo_tag(overrides)
        run_cfg_path = os.path.join(cfg_dir, f"{tag}.yaml")
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(merged, f)

        queue.append((tag, run_cfg_path, cpu_blocks[i]))

    running = []  # list of (tag, proc, log_file, log_path)
    finished = []  # list of (tag, returncode, log_path)

    def launch_next():
        tag, run_cfg_path, cpu_block = queue.pop(0)
        proc, log_file, log_path = launch(run_cfg_path, cpu_block, args, log_dir, tag)
        running.append((tag, proc, log_file, log_path))
        time.sleep(1)  # stagger run_name timestamps (1s resolution) so directories can't collide

    while queue and len(running) < args.max_parallel:
        launch_next()

    while running:
        time.sleep(5)
        still_running = []
        for tag, proc, log_file, log_path in running:
            code = proc.poll()
            if code is None:
                still_running.append((tag, proc, log_file, log_path))
                continue
            log_file.close()
            finished.append((tag, code, log_path))
            if code == 0:
                print(f"[sweep] finished {tag} (success) -> {log_path}")
            else:
                print(f"[sweep] FAILED {tag} (exit={code}) -> {log_path}")
        running = still_running

        while queue and len(running) < args.max_parallel:
            launch_next()

    n_failed = sum(1 for _, code, _ in finished if code != 0)
    print(f"[sweep] all {len(finished)} runs complete ({n_failed} failed). configs -> {cfg_dir}, logs -> {log_dir}")
    print("[sweep] compare with: tensorboard --logdir AdaptiveFPS/runs")


if __name__ == "__main__":
    main()
