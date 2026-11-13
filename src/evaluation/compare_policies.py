"""Compare saved baseline and shaped policies using only original LunarLander reward."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev

import gymnasium as gym
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = {
    "baseline": PROJECT_ROOT / "models" / "ppo_lunarlander_baseline.zip",
    "shaped": PROJECT_ROOT / "models" / "ppo_lunarlander_shaped.zip",
}
EVAL_SEEDS = tuple(range(1000, 1020))
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_policy(policy, model_path, seeds, render=False):
    """Run inference on a fresh original-reward environment; never load a reward wrapper."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    # Keep Gymnasium's standard TimeLimit; no custom reward wrapper is applied.
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    rows = []
    try:
        model = PPO.load(model_path, device="cpu")
        print(f"\nPolicy: {policy} | original LunarLander-v3 reward | deterministic inference")
        for episode, seed in enumerate(seeds, start=1):
            observation, info = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            rows.append({
                "policy": policy,
                "episode": episode,
                "seed": seed,
                "reward": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            print(
                f"Episode {episode:02d} | seed={seed} | reward={total_reward:.2f} | "
                f"steps={steps} | terminated={terminated} | truncated={truncated}"
            )
    finally:
        env.close()
    return rows


def summarize(rows):
    rewards = [row["reward"] for row in rows]
    lengths = [row["steps"] for row in rows]
    return {
        "episodes": len(rows),
        "mean_original_reward": mean(rewards),
        "reward_std_population": pstdev(rewards),
        "median_reward": median(rewards),
        "best_reward": max(rewards),
        "worst_reward": min(rewards),
        "mean_episode_length": mean(lengths),
        "median_episode_length": median(lengths),
        "min_episode_length": min(lengths),
        "max_episode_length": max(lengths),
        "episodes_reward_ge_200": sum(reward >= 200 for reward in rewards),
        "terminated_episodes": sum(row["terminated"] for row in rows),
        "truncated_episodes": sum(row["truncated"] for row in rows),
    }


def save_plot(results):
    """Save two compact paired-seed plots using the existing Matplotlib dependency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    try:
        for policy, rows in results.items():
            seeds = [row["seed"] for row in rows]
            axes[0].plot(seeds, [row["reward"] for row in rows], marker="o", label=policy)
            axes[1].plot(seeds, [row["steps"] for row in rows], marker="o", label=policy)
        axes[0].axhline(200, color="gray", linestyle="--", label="200 reward threshold")
        axes[0].set_ylabel("Original episode reward")
        axes[0].set_title("Baseline vs shaped PPO: same evaluation seeds, original reward")
        axes[1].set_ylabel("Episode length (steps)")
        axes[1].set_xlabel("Evaluation seed")
        axes[1].set_xticks(EVAL_SEEDS[::2])
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        path = RESULTS_DIR / "policy_comparison.png"
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)
    print(f"Plot saved to: {path}")


def compare_policies():
    results = {
        policy: evaluate_policy(policy, path, EVAL_SEEDS)
        for policy, path in MODEL_PATHS.items()
    }
    summaries = {policy: summarize(rows) for policy, rows in results.items()}

    # Pair explicitly by seed, rather than comparing unrelated episodes.
    baseline_by_seed = {row["seed"]: row for row in results["baseline"]}
    shaped_by_seed = {row["seed"]: row for row in results["shaped"]}
    paired = [
        {
            "seed": seed,
            "reward_difference": shaped_by_seed[seed]["reward"] - baseline_by_seed[seed]["reward"],
            "step_difference": shaped_by_seed[seed]["steps"] - baseline_by_seed[seed]["steps"],
        }
        for seed in EVAL_SEEDS
    ]
    differences = [row["reward_difference"] for row in paired]
    paired_summary = {
        "mean_reward_difference": mean(differences),
        "median_reward_difference": median(differences),
        "shaped_higher_reward_seeds": sum(value > 0 for value in differences),
        "baseline_higher_reward_seeds": sum(value < 0 for value in differences),
        "equal_reward_seeds": sum(value == 0 for value in differences),
        "mean_episode_length_difference": mean(row["step_difference"] for row in paired),
    }

    print("\nOriginal-reward comparison (reward standard deviation: population, ddof=0)")
    print(f"{'Metric':<29} {'baseline':>13} {'shaped':>13}")
    for metric in summaries["baseline"]:
        values = [summaries[policy][metric] for policy in MODEL_PATHS]
        formatted = [str(value) if isinstance(value, int) else f"{value:.2f}" for value in values]
        print(f"{metric:<29} {formatted[0]:>13} {formatted[1]:>13}")
    print("\nPaired differences: shaped minus baseline, both measured with original reward")
    for metric, value in paired_summary.items():
        formatted = str(value) if isinstance(value, int) else f"{value:.2f}"
        print(f"{metric}: {formatted}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "policy_comparison.csv"
    rows = results["baseline"] + results["shaped"]
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "environment": "LunarLander-v3",
            "reward": "original",
            "deterministic": True,
            "seeds": list(EVAL_SEEDS),
            "reward_std_convention": "population (ddof=0)",
        },
        "policies": summaries,
        "paired_summary": paired_summary,
        "paired_by_seed": paired,
    }
    summary_path = RESULTS_DIR / "policy_comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(f"Episode results saved to: {csv_path}")
    print(f"Summary and paired differences saved to: {summary_path}")
    save_plot(results)
    return summary


def run_visual_comparison(seed=1000):
    """Watch baseline then shaped on one shared seed, preserving benchmark files."""
    for policy, path in MODEL_PATHS.items():
        evaluate_policy(policy, path, [seed], render=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Watch both policies on one seed")
    parser.add_argument("--seed", type=int, default=None, help="Nonnegative seed for --render (default: 1000)")
    args = parser.parse_args()
    if args.seed is not None and not args.render:
        parser.error("--seed requires --render; the benchmark always uses seeds 1000-1019")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.render:
        run_visual_comparison(seed=1000 if args.seed is None else args.seed)
    else:
        compare_policies()
