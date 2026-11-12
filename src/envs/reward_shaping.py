"""Mild, transparent penalties added to LunarLander's original reward."""

import gymnasium as gym


# Costs 1 reward point per 100 steps: a modest incentive against lingering.
TIME_PENALTY = 0.01
# Small additional cost for large tilts; the original reward remains the main signal.
TILT_COEFFICIENT = 0.02
# Allow small corrective tilts (0.2 radians is about 11.5 degrees) without a penalty.
TILT_THRESHOLD = 0.2


class ShapedLunarLanderReward(gym.Wrapper):
    """Change only step rewards, without editing Gymnasium or changing transitions."""

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        # Index 4 is the lander's angle; use the newly observed state after the action.
        excess_tilt = max(0.0, abs(float(observation[4])) - TILT_THRESHOLD)
        tilt_penalty = TILT_COEFFICIENT * excess_tilt
        # Preserve all of the built-in reward, including crash/landing and engine terms.
        shaped_reward = float(original_reward) - TIME_PENALTY - tilt_penalty

        info = dict(info)
        info.update(
            original_reward=float(original_reward),
            time_penalty=TIME_PENALTY,
            tilt_penalty=tilt_penalty,
            shaped_reward=shaped_reward,
        )
        return observation, shaped_reward, terminated, truncated, info
