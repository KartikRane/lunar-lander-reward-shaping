"""Intentional failure experiment: reward stable states repeatedly, without a time cost."""

from math import hypot

import gymnasium as gym


ANGLE_THRESHOLD = 0.2  # Radians, about 11.5 degrees.
SPEED_THRESHOLD = 0.5  # Norm of the observation's scaled x/y velocities, not m/s.
STABILITY_BONUS = 1.0  # Deliberately repeatable every qualifying step.


class HackedLunarLanderReward(gym.Wrapper):
    """Add a stability bonus; an exploit is a hypothesis, not an assumed outcome."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_steps = 0
        self.bonus_steps = 0
        self.original_reward_total = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_steps = 0
        self.bonus_steps = 0
        self.original_reward_total = 0.0
        return observation, info

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        speed = hypot(float(observation[2]), float(observation[3]))
        upright_and_slow = abs(float(observation[4])) < ANGLE_THRESHOLD and speed < SPEED_THRESHOLD
        stability_bonus = STABILITY_BONUS if upright_and_slow else 0.0
        # Intentional flaw: reward occupying this state repeatedly, without requiring progress.
        hacked_reward = float(original_reward) + stability_bonus
        self.episode_steps += 1
        self.bonus_steps += int(upright_and_slow)
        self.original_reward_total += float(original_reward)

        info = dict(info)
        info.update(
            original_reward=float(original_reward),
            stability_bonus=stability_bonus,
            hacked_reward=hacked_reward,
            episode_original_reward=self.original_reward_total,
            episode_stability_bonus_steps=self.bonus_steps,
            episode_stability_bonus_fraction=self.bonus_steps / self.episode_steps,
        )
        return observation, hacked_reward, terminated, truncated, info
