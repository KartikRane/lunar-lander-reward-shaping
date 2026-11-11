"""Watch a random policy and inspect Gymnasium's interaction loop."""

import gymnasium as gym


ACTION_NAMES = {
    0: "Do nothing",
    1: "Fire left orientation engine",
    2: "Fire main engine",
    3: "Fire right orientation engine",
}

OBSERVATION_NAMES = (
    "x position",
    "y position",
    "x velocity",
    "y velocity",
    "angle",
    "angular velocity",
    "left leg ground contact",
    "right leg ground contact",
)


def run_random_agent(episodes=1, render_mode="human", log_interval=50):
    """Run without learning; use render_mode=None for a headless smoke check."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if log_interval < 0:
        raise ValueError("log_interval must be nonnegative (0 disables step logs)")

    env = gym.make("LunarLander-v3", render_mode=render_mode)
    try:
        print("Environment: LunarLander-v3")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print("Observation order (8 values): " + ", ".join(OBSERVATION_NAMES))
        print("Actions:")
        for action, name in ACTION_NAMES.items():
            print(f"  {action} -> {name}")

        for episode in range(1, episodes + 1):
            # reset starts a new episode and returns its first observation and info.
            observation, info = env.reset()
            print(f"\nEpisode {episode}")
            print(f"Initial observation ({len(observation)} values): {observation}")
            if info:
                print(f"Reset info: {info}")

            total_reward = 0.0
            steps = 0
            while True:
                # Sampling picks a valid random action; no policy is being learned.
                action = env.action_space.sample()
                # step returns the next state, immediate reward, end flags, and metadata.
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += float(reward)

                if log_interval and steps % log_interval == 0:
                    print(
                        f"Step {steps}: action={action} ({ACTION_NAMES[action]}), "
                        f"reward={reward:.2f}, total reward={total_reward:.2f}"
                    )

                # terminated: a task ending, such as a crash or settled landing.
                # truncated: an external cutoff, such as the environment time limit.
                if terminated or truncated:
                    print("Episode finished")
                    print(f"Total steps: {steps}")
                    print(f"Total reward: {total_reward:.2f}")
                    print(f"Terminated: {terminated}")
                    print(f"Truncated: {truncated}")
                    if info:
                        print(f"Final info: {info}")
                    break
    finally:
        # Release the renderer even if stepping fails or the run is interrupted.
        env.close()


if __name__ == "__main__":
    run_random_agent()
