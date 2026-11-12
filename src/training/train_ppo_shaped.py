"""Train a separate PPO model with mild time and excess-tilt penalties."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from src.envs.reward_shaping import ShapedLunarLanderReward


# Match the baseline seed and budget so reward design is the intended difference.
SEED = 42
TOTAL_TIMESTEPS = 500_000
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "ppo_lunarlander_shaped.zip"


def train_shaped(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, model_path=MODEL_PATH):
    """Train and save the shaped policy; return it with its environment closed."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    model_path = Path(model_path)
    baseline_path = MODEL_PATH.with_name("ppo_lunarlander_baseline.zip")
    if model_path.resolve() == baseline_path.resolve():
        raise ValueError("The shaped model must not overwrite the baseline model")

    # No rendering during training; the wrapper changes rewards, not the dynamics.
    env = ShapedLunarLanderReward(gym.make("LunarLander-v3"))
    try:
        # Keep the baseline's MlpPolicy, CPU device, and PPO defaults unchanged.
        # PPO also seeds the environment at its first training reset.
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")
        # Counts environment interactions; PPO completes full rollout batches.
        model.learn(total_timesteps=total_timesteps)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"Shaped model saved to: {model_path}")
        return model
    finally:
        env.close()


if __name__ == "__main__":
    train_shaped()
