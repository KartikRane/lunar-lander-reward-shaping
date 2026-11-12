"""Evaluate the saved PPO baseline without training or modifying rewards."""

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
from stable_baselines3 import PPO


EVAL_SEED = 1000
EVAL_EPISODES = 20
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_lunarlander_baseline.zip"
RESULTS_PATH = PROJECT_ROOT / "results" / "baseline_evaluation.csv"


def evaluate_baseline(
    episodes=EVAL_EPISODES,
    seed_base=EVAL_SEED,
    render=False,
    model_path=MODEL_PATH,
    results_path=RESULTS_PATH,
):
    """Return per-episode results; only headless runs write the benchmark CSV."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    episode_results = []
    try:
        # Loading with the fresh environment also checks observation/action compatibility.
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"Model: {model_path}")
        print(f"Mode: {'visual demo' if render else 'quantitative evaluation'}")
        print(f"Deterministic policy | episode seeds: {seed_base}-{seed_base + episodes - 1}")

        for episode_index in range(episodes):
            seed = seed_base + episode_index
            observation, info = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            while True:
                # Inference only: choose the policy's deterministic action, with no updates.
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                # A task ending or a time limit both require a reset before more steps.
                if terminated or truncated:
                    break

            result = {
                "episode": episode_index + 1,
                "seed": seed,
                "reward": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            episode_results.append(result)
            print(
                f"Episode {result['episode']:02d} | reward={total_reward:.2f} | "
                f"steps={steps} | terminated={terminated} | truncated={truncated}"
            )
    finally:
        env.close()

    rewards = [result["reward"] for result in episode_results]
    lengths = [result["steps"] for result in episode_results]
    print("\nEvaluation summary" if not render else "\nVisual demo summary")
    print(f"Number of episodes: {len(episode_results)}")
    print(f"Mean reward: {mean(rewards):.2f}")
    # Population standard deviation describes the evaluated set (ddof=0).
    print(f"Reward standard deviation (population): {pstdev(rewards):.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
    print(f"Worst episode reward: {min(rewards):.2f}")
    print(f"Mean episode length: {mean(lengths):.2f}")
    print(f"Minimum episode length: {min(lengths)}")
    print(f"Maximum episode length: {max(lengths)}")
    print(f"Episodes with reward >= 200: {sum(reward >= 200 for reward in rewards)}")

    if not render:
        # Preserve full reward precision in CSV; console output is rounded for readability.
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(episode_results[0]))
            writer.writeheader()
            writer.writerows(episode_results)
        print(f"Results saved to: {results_path}")
    return episode_results


def run_visual_demo():
    """Watch one deterministic episode without overwriting quantitative results."""
    return evaluate_baseline(episodes=1, render=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Watch one episode instead of running the benchmark")
    args = parser.parse_args()
    if args.render:
        run_visual_demo()
    else:
        evaluate_baseline()
