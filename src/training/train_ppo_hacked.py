"""Train a separate PPO policy with an intentionally repeatable stability bonus."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.reward_hacking import HackedLunarLanderReward


SEED = 42
TOTAL_TIMESTEPS = 500_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_lunarlander_hacked.zip"
MONITOR_PATH = PROJECT_ROOT / "results" / "ppo_lunarlander_hacked_monitor.csv"


def train_hacked(
    total_timesteps=TOTAL_TIMESTEPS,
    seed=SEED,
    model_path=MODEL_PATH,
    monitor_path=MONITOR_PATH,
):
    """Train the failure experiment, save its own model, and close the environment."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    model_path = Path(model_path)
    monitor_path = Path(monitor_path)
    for name in ("ppo_lunarlander_baseline.zip", "ppo_lunarlander_shaped.zip"):
        if model_path.resolve() == (PROJECT_ROOT / "models" / name).resolve():
            raise ValueError("The failure experiment must not overwrite existing models")
    monitor_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard SB3 monitoring exposes bonus frequency without a custom logger or callback.
    env = Monitor(
        HackedLunarLanderReward(gym.make("LunarLander-v3")),
        filename=str(monitor_path),
        info_keywords=(
            "episode_original_reward",
            "episode_stability_bonus_steps",
            "episode_stability_bonus_fraction",
        ),
    )
    try:
        # Match baseline/shaped PPO settings; start fresh rather than loading either model.
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")
        model.learn(total_timesteps=total_timesteps)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"Hacked-reward model saved to: {model_path}")
        print(f"Training episode diagnostics saved to: {monitor_path}")
        return model
    finally:
        env.close()


if __name__ == "__main__":
    train_hacked()
