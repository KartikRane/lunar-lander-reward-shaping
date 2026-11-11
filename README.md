# Lunar Lander Reward Shaping

Investigating reward shaping, reward hacking, and policy robustness in deep reinforcement learning using PPO and Gymnasium LunarLander-v3.

## Project Status

Work in progress. Current milestone: running and understanding a random policy in Gymnasium LunarLander-v3. No learning or reward shaping is implemented yet.

## Run the random agent

From the repository root in PowerShell, with a working Python virtual environment and the existing requirements installed:

```powershell
.\.venv\Scripts\python.exe -m src.random_agent
```

If dependencies are missing, install them with `python -m pip install -r requirements.txt` using your activated virtual environment.

A Lunar Lander window opens for one episode and closes when the episode ends. The terminal prints the observation/action spaces, the initial eight-value observation and its field order, readable action names, progress every 50 steps, and the final step count, total reward, and termination/truncation flags. A random agent will usually crash; this is an environment check, not a trained model or formal evaluation.

The observation contains position (x, y), velocity (x, y), angle, angular velocity, and two leg-contact indicators. Action names follow the [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

To watch more episodes, change the final call in `src/random_agent.py` to `run_random_agent(episodes=3)`. Set `log_interval=0` to suppress step logs. Press Ctrl+C in the terminal to interrupt; the environment is closed in a `finally` block.

`src/envs`, `src/training`, `src/evaluation`, and `src/utils` are empty package scaffolding. `configs`, `models`, `results`, and `videos` reserve space for future work; generated artifacts are ignored by Git.

## Planned Experiments

- Random-policy baseline
- PPO baseline
- Custom reward shaping
- Reward hacking experiments
- Corrected reward design
- Multi-seed evaluation
- Robustness testing under environmental disturbances
