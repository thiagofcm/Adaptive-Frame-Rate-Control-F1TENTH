from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

TENSORBOARD_LABELS = {
    "charts_episode_frame_count": {"title": "Ep Frame Count",                 "ylabel": "Ep Frames",       "xlabel": "Timesteps"},
    "charts_episodic_length":     {"title": "Episodic Length",                "ylabel": "Ep Length",       "xlabel": "Timesteps"},
    "charts_episodic_return":     {"title": "Episodic Reward",                "ylabel": "Ep Reward",       "xlabel": "Timesteps"},
    "charts_learning_rate":       {"title": "Learning Rate",                  "ylabel": "",                "xlabel": ""},
    "charts_mean_chosen_fps":     {"title": "Ep Mean Chosen FPS",             "ylabel": "Mean FPS",        "xlabel": "Timesteps"},
    "charts_mean_nav_reward":     {"title": "Mean Navigation Reward p/ Step", "ylabel": "Mean Nav Reward", "xlabel": "Timesteps"},
    "charts_SPS":                 {"title": "SPS",                           "ylabel": "",                "xlabel": ""},
    "losses_approx_kl":           {"title": "Approx. KL Divergence",          "ylabel": "",                "xlabel": ""},
    "losses_clipfrac":            {"title": "PPO Clip Fraction",              "ylabel": "",                "xlabel": ""},
    "losses_entropy":             {"title": "Policy Entropy",                 "ylabel": "",                "xlabel": ""},
    "losses_explained_variance":  {"title": "Explained Variance",             "ylabel": "",                "xlabel": ""},
    "losses_old_approx_kl":       {"title": "Approx. KL Divergence (Old)",    "ylabel": "",                "xlabel": ""},
    "losses_policy_loss":         {"title": "Policy Loss",                    "ylabel": "",                "xlabel": ""},
    "losses_value_loss":          {"title": "Value Function Loss",            "ylabel": "",                "xlabel": ""},
}


def plot_tensorboard(csv_dir, output_dir, smooth_rew=0.6, smooth_mean_fps=0.6, smooth_frame_count=0.6):
    """Plots each exported tensorboard CSV individually, mirroring plot_csv.py's style."""
    # Imported here (not at module scope) so plot_csv.py's global style/rcParams
    # override only takes effect after export_tensorboard()'s grid plot is saved.
    from plot_csv import smooth

    smooth_weights = {
        "charts_episodic_return":     smooth_rew,
        "charts_mean_chosen_fps":     smooth_mean_fps,
        "charts_episode_frame_count": smooth_frame_count,
    }

    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue

        stem = os.path.splitext(fname)[0]
        labels = TENSORBOARD_LABELS.get(stem)
        if labels is None:
            continue

        df = pd.read_csv(os.path.join(csv_dir, fname))
        step_col  = next((c for c in df.columns if c.lower() == "step"),  None)
        value_col = next((c for c in df.columns if c.lower() == "value"), None)
        if step_col is None or value_col is None:
            continue

        steps  = df[step_col].values
        values = df[value_col].values
        weight = smooth_weights.get(stem)

        plt.figure(figsize=(18, 15))
        if weight is not None:
            smoothed = smooth(values, weight=weight)
            plt.plot(steps, values,   alpha=0.3, color="steelblue", linewidth=1.0, label="Raw")
            plt.plot(steps, smoothed, alpha=1.0, color="steelblue", linewidth=2.0, label=f"Smoothed (w={weight})")
        else:
            plt.plot(steps, values, alpha=1.0, color="steelblue", linewidth=2.0)

        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

        plt.xlabel(labels["xlabel"])
        plt.ylabel(labels["ylabel"])
        plt.title(labels["title"])
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )

        if stem == "charts_mean_chosen_fps":
            plt.ylim(0, 12)
            plt.yticks([1, 2, 5, 10])
            for fps in (1, 2, 5, 10):
                plt.axhline(y=fps, color="gray", linestyle=":", alpha=0.7)

        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{stem}_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
        plt.close()


def export_tensorboard(log_dir, output_dir):
    csv_output = os.path.join(output_dir, "csv")
    os.makedirs(csv_output, exist_ok=True)

    ea = EventAccumulator(log_dir)
    ea.Reload()

    # ── Export each scalar tag to its own CSV ──────────────────────────────
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step":  [e.step  for e in events],
            "value": [e.value for e in events],
        })
        # e.g. "charts/episodic_return" → "charts_episodic_return.csv"
        fname = tag.replace("/", "_") + ".csv"
        df.to_csv(os.path.join(csv_output, fname), index=False)
        print(f"Saved → {fname}")

    # ── Plot all scalars ───────────────────────────────────────────────────
    tags = ea.Tags()["scalars"]
    n    = len(tags)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i, tag in enumerate(tags):
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        axes[i].plot(steps, values)
        axes[i].set_title(tag)
        axes[i].set_xlabel("step")

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_scalars.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",              type=str,   required=True)
    parser.add_argument("--smooth_rew",          type=float, default=0.6, help="Smoothing weight for episodic reward")
    parser.add_argument("--smooth_mean_fps",     type=float, default=0.6, help="Smoothing weight for mean chosen FPS")
    parser.add_argument("--smooth_frame_count",  type=float, default=0.6, help="Smoothing weight for episode frame count")
    args = parser.parse_args()

    log_dir    = args.logdir
    output_dir = os.path.join(log_dir,"training_plots")
    os.makedirs(output_dir, exist_ok=True)
    export_tensorboard(log_dir, output_dir)

    csv_dir = os.path.join(output_dir, "csv")
    plot_tensorboard(csv_dir, output_dir,
                      smooth_rew=args.smooth_rew,
                      smooth_mean_fps=args.smooth_mean_fps,
                      smooth_frame_count=args.smooth_frame_count)