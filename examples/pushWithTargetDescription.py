import gymnasium as gym

import panda_gym

env = gym.make("PandaPush-v3", render_mode="human")

observation, info = env.reset()

for _ in range(50000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
