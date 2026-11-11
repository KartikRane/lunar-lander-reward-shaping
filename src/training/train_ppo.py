"""Train the first PPO baseline using LunarLander's original reward."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


SEED = 42
# An initial learning budget, not a guarantee of solving this stochastic task.
TOTAL_TIMESTEPS = 500_000
MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "ppo_lunarlander_baseline.zip"
)


def train_baseline(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, model_path=MODEL_PATH):
    """Train and save one baseline; return the model with its environment closed."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    model_path = Path(model_path)

    # Rendering is disabled to avoid slowing training; the original reward is used.
    env = gym.make("LunarLander-v3")
    try:
        # MlpPolicy uses fully connected networks for action selection and value estimates.
        # PPO's seed also seeds the environment at its first training reset.
        # SB3 automatically adds episode monitoring for its normal reward/length logs.
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")

        # PPO collects experience and updates the policy with constrained policy changes.
        # Timesteps count environment interactions, not episodes or gradient updates.
        # PPO completes full rollouts, so the actual count can exceed this budget slightly.
        model.learn(total_timesteps=total_timesteps)

        # Save the learned policy so a later milestone can load it without retraining.
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"Baseline model saved to: {model_path}")
        return model
    finally:
        env.close()


if __name__ == "__main__":
    train_baseline()
