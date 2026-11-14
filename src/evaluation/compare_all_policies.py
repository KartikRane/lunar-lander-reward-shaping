"""Compare three saved policies on original reward and inspect hacked-training diagnostics."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from src.evaluation.compare_policies import EVAL_SEEDS, PROJECT_ROOT, evaluate_policy, summarize


MODEL_PATHS = {
    policy: PROJECT_ROOT / "models" / f"ppo_lunarlander_{policy}.zip"
    for policy in ("baseline", "shaped", "hacked")
}
RESULTS_DIR = PROJECT_ROOT / "results"
MONITOR_PATH = RESULTS_DIR / "ppo_lunarlander_hacked_monitor.csv"
TRAINING_WINDOW = 20


def check_inputs(include_monitor=True):
    paths = list(MODEL_PATHS.values())
    if include_monitor:
        paths.append(MONITOR_PATH)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required experiment files are missing:\n" + "\n".join(missing))


def paired_comparison(left, right):
    """Return left-minus-right differences, matched explicitly by evaluation seed."""
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    if (len(left_by_seed) != len(left) or len(right_by_seed) != len(right)
            or left_by_seed.keys() != right_by_seed.keys() or not left):
        raise ValueError("Paired rows must have matching, unique, nonempty seed sets")
    paired = [
        {
            "seed": seed,
            "reward_difference": left_by_seed[seed]["reward"] - right_by_seed[seed]["reward"],
            "step_difference": left_by_seed[seed]["steps"] - right_by_seed[seed]["steps"],
        }
        for seed in sorted(left_by_seed)
    ]
    rewards = [row["reward_difference"] for row in paired]
    return {
        "mean_reward_difference": mean(rewards),
        "median_reward_difference": median(rewards),
        "higher_reward_seed_count": sum(value > 0 for value in rewards),
        "lower_reward_seed_count": sum(value < 0 for value in rewards),
        "ties": sum(value == 0 for value in rewards),
        "mean_episode_length_difference": mean(row["step_difference"] for row in paired),
        "per_seed": paired,
    }


def analyze_hacked_training(path=MONITOR_PATH, window=TRAINING_WINDOW):
    """Read completed training episodes; these are not deterministic evaluation episodes."""
    if window < 1:
        raise ValueError("Training window must be at least 1")
    with Path(path).open(encoding="utf-8-sig", newline="") as input_file:
        # SB3 Monitor writes one JSON metadata comment before the CSV header.
        rows = list(csv.DictReader(line for line in input_file if line.strip() and not line.startswith("#")))
    if not rows:
        raise ValueError("The hacked Monitor CSV contains no completed episodes")
    required = ("r", "l", "episode_original_reward", "episode_stability_bonus_steps",
                "episode_stability_bonus_fraction")
    if not all(field in rows[0] for field in required):
        raise ValueError(f"Monitor CSV must contain these columns: {required}")
    final = [{field: float(row[field]) for field in required} for row in rows[-window:]]
    modified = mean(row["r"] for row in final)
    bonus_reward = mean(row["r"] - row["episode_original_reward"] for row in final)
    return {
        "completed_training_episodes": len(rows),
        "window_episodes": len(final),
        "mean_modified_training_reward": modified,
        "mean_original_training_reward": mean(row["episode_original_reward"] for row in final),
        "mean_episode_length": mean(row["l"] for row in final),
        "mean_bonus_eligible_steps": mean(row["episode_stability_bonus_steps"] for row in final),
        "mean_bonus_fraction": mean(row["episode_stability_bonus_fraction"] for row in final),
        "mean_bonus_reward_contribution": bonus_reward,
        "bonus_share_of_modified_reward": bonus_reward / modified if modified > 0 else None,
        # The known experiment gives exactly +1 for each qualifying step; Monitor rounds r.
        "max_bonus_accounting_error": max(
            abs(row["r"] - row["episode_original_reward"] - row["episode_stability_bonus_steps"])
            for row in final
        ),
    }


def save_plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    try:
        for policy, rows in results.items():
            seeds = [row["seed"] for row in rows]
            axes[0].plot(seeds, [row["reward"] for row in rows], marker="o", label=policy)
            axes[1].plot(seeds, [row["steps"] for row in rows], marker="o", label=policy)
        axes[0].set_title("Three PPO policies: original reward, same evaluation seeds")
        axes[0].set_ylabel("Original episode reward")
        axes[0].axhline(200, color="gray", linestyle="--", label="200 reward threshold")
        axes[1].set_ylabel("Episode length (steps)")
        axes[1].set_xlabel("Evaluation seed")
        axes[1].set_xticks(EVAL_SEEDS[::2])
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        path = RESULTS_DIR / "all_policy_comparison.png"
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)
    print(f"Plot saved to: {path}")


def compare_all_policies():
    # Fail before evaluating or writing outputs if either the model or diagnostics is absent.
    check_inputs()
    diagnostics = analyze_hacked_training()
    # Reuse the original-reward evaluator; it applies no custom reward wrappers.
    results = {
        policy: evaluate_policy(policy, path, EVAL_SEEDS)
        for policy, path in MODEL_PATHS.items()
    }
    summaries = {policy: summarize(rows) for policy, rows in results.items()}
    pairs = {
        f"{left}_minus_{right}": paired_comparison(results[left], results[right])
        for left, right in (("shaped", "baseline"), ("hacked", "baseline"), ("hacked", "shaped"))
    }
    print("\nOriginal-reward benchmark (population reward standard deviation, ddof=0)")
    print(f"{'Metric':<29}" + "".join(f"{policy:>14}" for policy in MODEL_PATHS))
    for metric in summaries["baseline"]:
        values = [summaries[policy][metric] for policy in MODEL_PATHS]
        formatted = [str(value) if isinstance(value, int) else f"{value:.2f}" for value in values]
        print(f"{metric:<29}" + "".join(f"{value:>14}" for value in formatted))
    for name, comparison in pairs.items():
        print(f"\nPaired comparison: {name}")
        for metric, value in comparison.items():
            if metric != "per_seed":
                print(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")
    print(f"\nHacked training diagnostics: final {diagnostics['window_episodes']} completed episodes")
    for metric, value in diagnostics.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    # Suggest examples using the complete benchmark, rather than cherry-picking evaluation.
    hacked_vs_shaped = pairs["hacked_minus_shaped"]["per_seed"]
    largest_delay = max(hacked_vs_shaped, key=lambda row: row["step_difference"])
    largest_reward_gap = min(hacked_vs_shaped, key=lambda row: row["reward_difference"])
    suggested = {
        "largest_hacked_minus_shaped_step_difference": largest_delay,
        "lowest_hacked_minus_shaped_reward_difference": largest_reward_gap,
    }
    print("\nSeeds for optional visual inspection:")
    for reason, row in suggested.items():
        print(f"{reason}: seed={row['seed']}, reward difference={row['reward_difference']:.2f}, "
              f"step difference={row['step_difference']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row for policy_rows in results.values() for row in policy_rows]
    csv_path = RESULTS_DIR / "all_policy_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {"environment": "LunarLander-v3", "reward": "original",
                     "deterministic": True, "seeds": list(EVAL_SEEDS),
                     "reward_std_convention": "population (ddof=0)"},
        "policies": summaries,
        "pairwise": pairs,
        "hacked_training_diagnostics": diagnostics,
        "suggested_visual_seeds": suggested,
    }
    summary_path = RESULTS_DIR / "all_policy_comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(f"Episode results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    save_plot(results)
    return summary


def run_visual_comparison(seed=1000):
    """Run baseline, shaped, then hacked on one seed; write no benchmark files."""
    check_inputs(include_monitor=False)
    for policy, path in MODEL_PATHS.items():
        evaluate_policy(policy, path, [seed], render=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Watch all three policies on one seed")
    parser.add_argument("--seed", type=int, default=None, help="Seed for --render (default: 1000)")
    args = parser.parse_args()
    if args.seed is not None and (not args.render or args.seed < 0):
        parser.error("--seed requires --render and must be nonnegative")
    if args.render:
        run_visual_comparison(seed=1000 if args.seed is None else args.seed)
    else:
        compare_all_policies()
