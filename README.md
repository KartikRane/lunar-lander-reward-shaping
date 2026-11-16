# Lunar Lander Reward Shaping

Investigating reward shaping, reward hacking, and policy robustness in deep reinforcement learning using PPO and Gymnasium LunarLander-v3.

## Project Status

Work in progress. The random-agent check, PPO baseline training, and separate deterministic evaluation pipeline are available. The baseline uses the original Gymnasium LunarLander-v3 reward. A first separate training experiment adds mild time and excess-tilt penalties. A controlled comparison evaluates both saved policies with the original reward on the same episode seeds.

## Run the random agent

From the repository root in PowerShell, with a working Python virtual environment and the existing requirements installed:

```powershell
.\.venv\Scripts\python.exe -m src.random_agent
```

If dependencies are missing, install them with `python -m pip install -r requirements.txt` using your activated virtual environment.

A Lunar Lander window opens for one episode and closes when the episode ends. The terminal prints the observation/action spaces, the initial eight-value observation and its field order, readable action names, progress every 50 steps, and the final step count, total reward, and termination/truncation flags. A random agent will usually crash; this is an environment check, not a trained model or formal evaluation.

The observation contains position (x, y), velocity (x, y), angle, angular velocity, and two leg-contact indicators. Action names follow the [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

To watch more episodes, change the final call in `src/random_agent.py` to `run_random_agent(episodes=3)`. Set `log_interval=0` to suppress step logs. Press Ctrl+C in the terminal to interrupt; the environment is closed in a `finally` block.

`src/envs` contains the reward-shaping wrapper, `src/training` contains separate baseline/shaped PPO scripts, and `src/evaluation` contains baseline evaluation and a controlled baseline-vs-shaped comparison. `src/utils` remains empty package scaffolding. `configs`, `models`, `results`, and `videos` reserve space for future work; generated artifacts are ignored by Git.

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

After training completes, the script saves `models/ppo_lunarlander_baseline.zip` and prints its absolute path. Running training again starts a fresh model and overwrites that file on completion. Generated models remain ignored by Git. Generated trained models and evaluation CSV files are local artifacts and are not committed.

## Evaluate the saved PPO baseline

After training has produced `models/ppo_lunarlander_baseline.zip`, run this from the repository root using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_ppo
```

With the virtual environment activated, the equivalent command is `python -m src.evaluation.evaluate_ppo`.

The benchmark runs 20 fresh, headless episodes using deterministic policy inference and reset seeds 1000 through 1019. It performs no learning and uses the original environment reward. Each episode prints its total reward, step count, and termination/truncation flags, followed by reward and episode-length summary statistics. Reward standard deviation uses the population convention (`ddof=0`). The count labeled `Episodes with reward >= 200` is a reward threshold count; it does not identify successful landings.

Full-precision per-episode results are saved to `results/baseline_evaluation.csv` with columns `episode`, `seed`, `reward`, `steps`, `terminated`, and `truncated`. Repeating the benchmark overwrites this CSV. Paths are resolved from the source file's location, so the repository can be cloned elsewhere. Fixed reset seeds and deterministic inference make runs repeatable within the same software/hardware setup; results may vary across setups. The benchmark describes this model on this fixed episode set, not general robustness.

To watch one episode in a Lunar Lander window:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_ppo --render
```

The visual demo uses seed 1000 and does not overwrite the benchmark CSV. Its window closes after the episode; the main 20-episode benchmark never renders. Both modes load the model without saving or modifying it. See the [SB3 inference API](https://stable-baselines3.readthedocs.io/en/master/modules/base.html) for `load` and deterministic `predict` behavior.

## First reward-shaping experiment

`src/envs/reward_shaping.py` uses a [Gymnasium wrapper](https://gymnasium.farama.org/api/wrappers/) to preserve the original reward while adding two small penalties:

```text
shaped_reward = original_reward - 0.01 - 0.02 * max(0, abs(angle) - 0.2)
```

The angle is observation index 4, in radians, taken after each action. `TIME_PENALTY = 0.01` discourages lingering (1 point per 100 steps). `TILT_COEFFICIENT = 0.02` gently discourages excessive tilt, with `TILT_THRESHOLD = 0.2` radians (about 11.5 degrees) allowing small corrective tilts. For example, an angle of 1 radian adds a 0.016-point tilt penalty per step. These conservative coefficients are an initial design choice, not tuned values or a guarantee of improvement. The additional penalties apply to every step, including the final step.

The wrapper leaves observations, environment dynamics, and termination/truncation flags unchanged. Each step's `info` includes `original_reward`, `time_penalty`, `tilt_penalty`, and `shaped_reward` for inspection, without printing per-step logs.

Train the separate shaped policy from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_ppo_shaped
```

With the virtual environment activated, use `python -m src.training.train_ppo_shaped`. The script uses the same seed (42), 500,000-step budget, CPU execution, `MlpPolicy`, and PPO defaults as the baseline. The completed model is saved to `models/ppo_lunarlander_shaped.zip`. Repeating this command starts fresh and overwrites only that shaped-model output on completion.

Training's `ep_rew_mean` now measures shaped reward and must not be compared directly with the baseline's original-reward statistics. Both policies must be compared on original-reward LunarLander. The baseline evaluator still targets only the baseline; the separate comparison below measures both saved models without retraining.

## Compare baseline and shaped policies

Both saved models are evaluated on fresh original-reward `LunarLander-v3` environments, with no custom reward wrapper. Gymnasium's standard time limit is retained. The benchmark uses deterministic inference on the same 20 reset seeds (1000-1019) for both policies, with no rendering or learning.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare_policies
```

This prints each episode's results, aggregate reward/episode-length statistics, and paired differences (shaped minus baseline). Reward standard deviation is the population standard deviation (`ddof=0`), matching the baseline evaluator. Results are saved to:

- `results/policy_comparison.csv`: 40 rows, one per policy and episode.
- `results/policy_comparison_summary.json`: aggregate metrics, protocol, and per-seed paired differences.
- `results/policy_comparison.png`: per-seed original rewards and episode lengths.

Repeating the benchmark replaces only these comparison outputs. It leaves both trained models and `results/baseline_evaluation.csv` unchanged. To watch the baseline followed by the shaped policy on the same selected seed:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare_policies --render --seed 1000
```

Visual mode writes no result files. Shorter episodes and higher original rewards are separate observations; episode length alone does not establish better behavior. Reward >= 200 is a threshold count, not a direct landing-success measure. This is a comparison of two single-training-seed policies on a fixed set of episodes, not evidence of general robustness or a final project conclusion.

## Intentional failure experiment: repeatable stability bonus

Hypothesis: rewarding stability every timestep may create an unintended incentive to remain stable without completing the landing. This experiment deliberately adds no time penalty or progress requirement, and does not change the first shaped reward.

```text
speed = sqrt(observation[2]**2 + observation[3]**2)
stability_bonus = 1.0 if abs(observation[4]) < 0.2 and speed < 0.5 else 0.0
hacked_reward = original_reward + stability_bonus
```

`ANGLE_THRESHOLD = 0.2` is in radians (about 11.5 degrees); `SPEED_THRESHOLD = 0.5` uses the scaled velocity values in the observation, not physical metres per second. The strict conditions are checked on the observation returned after each action. The `+1.0` bonus can be collected repeatedly, including on a final step if it qualifies. Observations, dynamics, and end flags are preserved. Per-step `info` exposes `original_reward`, `stability_bonus`, and `hacked_reward`, plus episode-level original reward and bonus counts/fraction.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_ppo_hacked
```

With an activated virtual environment, use `python -m src.training.train_ppo_hacked`. Training starts a fresh `MlpPolicy` with the same seed (42), 500,000-step budget, CPU execution, and PPO defaults as the earlier experiments. It saves `models/ppo_lunarlander_hacked.zip`, leaving baseline/shaped models untouched.

Normal SB3 output reports the modified training reward. The standard Monitor file `results/ppo_lunarlander_hacked_monitor.csv` also records each completed training episode's original reward, number of bonus-eligible steps, and bonus fraction. The CSV begins with SB3's metadata comment, followed by a header; `r` is modified episode reward and `l` is episode length. Repeating training replaces only this experiment's model and Monitor output.

High modified reward or long episodes alone do not establish reward hacking. Hovering, delayed landing, inefficient trajectories, or no clear exploit are all possible outcomes. A later evaluation must measure this policy using the original unwrapped LunarLander reward and inspect behavior. No evaluation or anti-hacking correction is part of this step.

## Evaluate and diagnose the stability-bonus policy

The three-policy benchmark evaluates baseline, shaped, and hacked policies on fresh original-reward `LunarLander-v3` environments. It reuses the existing deterministic evaluator, applies no custom reward wrapper, retains Gymnasium's standard time limit, and uses reset seeds 1000-1019 for every policy. There is no learning or model saving.

All three model ZIPs and `results/ppo_lunarlander_hacked_monitor.csv` must exist before running:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare_all_policies
```

Outputs are `results/all_policy_comparison.csv` (60 episode rows), `results/all_policy_comparison_summary.json` (aggregates, three paired comparisons, final training diagnostics, and suggested visual seeds), and `results/all_policy_comparison.png` (original reward and episode length by seed). Population reward standard deviation (`ddof=0`) is used for every policy. Existing two-policy outputs, baseline CSV, Monitor data, and model files remain untouched. Rerunning replaces only the three new comparison outputs.

The diagnostics reader skips SB3's Monitor metadata comment and summarizes the final 20 completed training episodes, or all available episodes if fewer than 20 exist. It separates modified training reward from original training reward, reports bonus frequency, and checks how closely their difference matches the number of +1 bonus steps. These training episodes are stochastic training samples, not the deterministic evaluation episodes, and high modified reward alone does not prove an exploit.

Watch baseline, then shaped, then hacked on one shared seed:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare_all_policies --render --seed 1000
```

Visual mode writes no result files. Inspect the hacked policy for delayed completion, hovering, oscillation, crashes, normal landing, or reaching the time limit. Use original rewards, episode lengths, training diagnostics, and observed behavior together; avoid inferring behavior from a reward threshold or episode length alone. The suggested seeds supplement the full 20-seed evidence. No corrected reward or automatic reward-hacking conclusion is implemented here.

## Step 8: corrected-reward hypothesis

The WSL experiment reported a failure with the repeated stability bonus: the hacked policy collected the bonus while hovering rather than completing the landing. Over its final 20 completed training episodes, the reported means were 953.11 modified reward versus -24.34 original reward, 1,000 steps, and 977.45 bonus-eligible steps (97.75%). Original-reward evaluation reported a mean of -9.87 with all 20 episodes reaching the time limit. Visual inspection on seed 1005 confirmed hovering until that limit. These are supplied WSL observations, not results regenerated by this implementation step.

The flaw was rewarding occupancy of a stable state every timestep, without requiring progress. The new `CorrectedLunarLanderReward` keeps the original reward as the base and uses:

```text
descent_progress = max(0, previous_y - current_y)
corrected_reward = original_reward
                   - 0.01
                   - 0.02 * max(0, abs(angle) - 0.2)
                   + 0.1 * descent_progress
```

The small time penalty discourages unnecessarily long episodes. The excess-tilt penalty allows normal steering within 0.2 radians, while discouraging larger tilts. The small descent-progress bonus rewards a change in observation y toward the ground; staying upright, slow, or stationary gives no positive state bonus. Position y and angle come from observation indices 1 and 4. Previous y is initialized at every reset and updated after every step, including ascents and terminal steps.

The wrapper exposes `original_reward`, `time_penalty`, `tilt_penalty`, `descent_progress`, `progress_bonus`, and `corrected_reward` in `info`, together with episode totals for the original reward and each added component. Observations, dynamics, and termination/truncation flags are preserved. The descent term uses the environment's scaled y coordinate; it does not measure horizontal alignment or safe landing. Positive-only descent progress can accumulate over climb/descent cycles, so this is not a guarantee against every exploit.

Run training manually in WSL from the repository root, using your existing WSL virtual environment:

```bash
source .venv/bin/activate
python -m src.training.train_ppo_corrected
```

The script starts a fresh PPO `MlpPolicy` using seed 42, a 500,000-step budget, CPU execution, and the same PPO defaults as earlier experiments. No training starts on import. It saves `models/ppo_lunarlander_corrected.zip` and standard Monitor output to `results/ppo_lunarlander_corrected_monitor.csv`. The Monitor includes episode totals for the original reward, time penalty, tilt penalty, and progress bonus; its `r` column is corrected reward. Safety checks protect the three earlier model names and Monitor outputs. Rerunning corrected training replaces only this experiment's default outputs.

The corrected reward is a hypothesis designed to remove the hovering incentive. Its effectiveness must be tested empirically. This implementation step does not train or evaluate it. A later comparison must evaluate the corrected policy under the original LunarLander reward; corrected training returns are not directly comparable to that benchmark.

Source code is synchronized through GitHub, while trained models, Monitor CSVs, and generated evaluation artifacts remain Git-ignored and may exist only in WSL. Their absence on Windows does not imply that a WSL experiment has not run, and no existing ignored artifact is required to implement or import this step.

## Planned Experiments

- Random-policy baseline
- PPO baseline
- Custom reward shaping
- Reward hacking experiments
- Corrected reward design
- Multi-seed evaluation
- Robustness testing under environmental disturbances
