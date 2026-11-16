"""Correction hypothesis: small costs and descent progress, without a state-occupancy bonus."""

import gymnasium as gym


TIME_PENALTY = 0.01
TILT_THRESHOLD = 0.2
TILT_COEFFICIENT = 0.02
DESCENT_PROGRESS_COEFFICIENT = 0.1


class CorrectedLunarLanderReward(gym.Wrapper):
    """Preserve the original reward and transitions while adding small progress-based shaping."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_y = None
        self.episode_original_reward = 0.0
        self.episode_time_penalty = 0.0
        self.episode_tilt_penalty = 0.0
        self.episode_progress_bonus = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Seed the progress comparison from this episode, never from a previous episode.
        self.previous_y = float(observation[1])
        self.episode_original_reward = 0.0
        self.episode_time_penalty = 0.0
        self.episode_tilt_penalty = 0.0
        self.episode_progress_bonus = 0.0
        return observation, info

    def step(self, action):
        if self.previous_y is None:
            raise RuntimeError("Call reset() before stepping the corrected reward wrapper")
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        current_y = float(observation[1])
        excess_tilt = max(0.0, abs(float(observation[4])) - TILT_THRESHOLD)
        tilt_penalty = TILT_COEFFICIENT * excess_tilt
        # Observation y decreases during descent; stationary hovering earns no progress bonus.
        descent_progress = max(0.0, self.previous_y - current_y)
        progress_bonus = DESCENT_PROGRESS_COEFFICIENT * descent_progress
        corrected_reward = float(original_reward) - TIME_PENALTY - tilt_penalty + progress_bonus
        # Update even when ascending, so the next comparison is between consecutive states.
        self.previous_y = current_y

        self.episode_original_reward += float(original_reward)
        self.episode_time_penalty += TIME_PENALTY
        self.episode_tilt_penalty += tilt_penalty
        self.episode_progress_bonus += progress_bonus
        info = dict(info)
        info.update(
            original_reward=float(original_reward),
            time_penalty=TIME_PENALTY,
            tilt_penalty=tilt_penalty,
            descent_progress=descent_progress,
            progress_bonus=progress_bonus,
            corrected_reward=corrected_reward,
            episode_original_reward=self.episode_original_reward,
            episode_time_penalty=self.episode_time_penalty,
            episode_tilt_penalty=self.episode_tilt_penalty,
            episode_progress_bonus=self.episode_progress_bonus,
        )
        return observation, corrected_reward, terminated, truncated, info
