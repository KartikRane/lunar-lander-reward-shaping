"""Manually train the fourth PPO experiment: a corrected-reward hypothesis."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.corrected_reward import CorrectedLunarLanderReward


SEED = 42
TOTAL_TIMESTEPS = 500_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_lunarlander_corrected.zip"
MONITOR_PATH = PROJECT_ROOT / "results" / "ppo_lunarlander_corrected_monitor.csv"


def train_corrected(
    total_timesteps=TOTAL_TIMESTEPS,
    seed=SEED,
    model_path=MODEL_PATH,
    monitor_path=MONITOR_PATH,
):
    """Start fresh with baseline PPO settings; save only this experiment's artifacts."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    model_path = Path(model_path)
    # Normalize the filename before safety checks, matching SB3's .zip output suffix.
    if model_path.suffix != ".zip":
        model_path = Path(f"{model_path}.zip")
    monitor_path = Path(monitor_path)
    if not str(monitor_path).endswith("monitor.csv"):
        monitor_path = Path(f"{monitor_path}.monitor.csv")
    for policy in ("baseline", "shaped", "hacked"):
        protected_model = PROJECT_ROOT / "models" / f"ppo_lunarlander_{policy}.zip"
        protected_monitor = PROJECT_ROOT / "results" / f"ppo_lunarlander_{policy}_monitor.csv"
        if model_path.resolve() == protected_model.resolve():
            raise ValueError("Corrected training must not overwrite a previous model")
        if monitor_path.resolve() == protected_monitor.resolve():
            raise ValueError("Corrected training must not overwrite previous Monitor data")

    print("Training corrected-reward PPO agent")
    print("Base objective: original LunarLander reward")
    print("Additional shaping: small time penalty, excess-tilt penalty, descent-progress bonus")
    print("Purpose: test removal of the hovering incentive observed with the repeatable stability bonus")
    print("This reward is a hypothesis; effectiveness requires later original-reward evaluation.")
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = CorrectedLunarLanderReward(gym.make("LunarLander-v3"))
    try:
        # Monitor records corrected return as r and the component totals on each episode ending.
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "episode_original_reward",
                "episode_time_penalty",
                "episode_tilt_penalty",
                "episode_progress_bonus",
            ),
        )
        # Keep policy, device, seed and all PPO defaults consistent with previous experiments.
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")
        model.learn(total_timesteps=total_timesteps)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"Corrected model saved to: {model_path}")
        print(f"Training episode diagnostics saved to: {monitor_path}")
        return model
    finally:
        env.close()


if __name__ == "__main__":
    train_corrected()
