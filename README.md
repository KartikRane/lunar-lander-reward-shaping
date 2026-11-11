# Lunar Lander Reward Shaping

Investigating reward shaping, reward hacking, and policy robustness in deep reinforcement learning using PPO and Gymnasium LunarLander-v3.

## Project Status

Work in progress. The random-agent environment check is available. Current milestone: a PPO baseline training pipeline using the original Gymnasium LunarLander-v3 reward. Performance has not been evaluated; reward shaping is not implemented.

## Run the random agent

From the repository root in PowerShell, with a working Python virtual environment and the existing requirements installed:

```powershell
.\.venv\Scripts\python.exe -m src.random_agent
```

If dependencies are missing, install them with `python -m pip install -r requirements.txt` using your activated virtual environment.

A Lunar Lander window opens for one episode and closes when the episode ends. The terminal prints the observation/action spaces, the initial eight-value observation and its field order, readable action names, progress every 50 steps, and the final step count, total reward, and termination/truncation flags. A random agent will usually crash; this is an environment check, not a trained model or formal evaluation.

The observation contains position (x, y), velocity (x, y), angle, angular velocity, and two leg-contact indicators. Action names follow the [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

To watch more episodes, change the final call in `src/random_agent.py` to `run_random_agent(episodes=3)`. Set `log_interval=0` to suppress step logs. Press Ctrl+C in the terminal to interrupt; the environment is closed in a `finally` block.

`src/envs`, `src/evaluation`, and `src/utils` are empty package scaffolding; `src/training` contains the PPO baseline script. `configs`, `models`, `results`, and `videos` reserve space for future work; generated artifacts are ignored by Git.

## Train the PPO baseline

From the repository root in PowerShell, activate the project's virtual environment and start training:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.training.train_ppo
```

You can also run it without activation:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_ppo
```

Training uses the original environment reward, `MlpPolicy` (fully connected policy/value networks), seed 42, CPU execution, and otherwise Stable-Baselines3 PPO defaults. There is no rendering window during training. `TOTAL_TIMESTEPS = 500_000` near the top of `src/training/train_ppo.py` is the initial interaction budget; edit it for later runs. PPO finishes complete rollouts, so the actual number of steps may slightly exceed the requested budget. This single seeded run does not guarantee reproducibility across platforms or successful landings.

Stable-Baselines3 prints progress tables: `ep_rew_mean` is the mean total reward of recently completed training episodes; `ep_len_mean` is their mean length in environment steps (by default, both use up to the last 100 episodes). `total_timesteps` counts environment interactions, and `fps` reports training throughput. These training statistics are not a separate evaluation. See the [SB3 logger documentation](https://stable-baselines3.readthedocs.io/en/stable/common/logger.html) for the other optimizer metrics.

After training completes, the script saves `models/ppo_lunarlander_baseline.zip` and prints its absolute path. Running training again starts a fresh model and overwrites that file on completion. Generated models remain ignored by Git. No trained baseline or performance results are included yet.

## Planned Experiments

- Random-policy baseline
- PPO baseline
- Custom reward shaping
- Reward hacking experiments
- Corrected reward design
- Multi-seed evaluation
- Robustness testing under environmental disturbances
