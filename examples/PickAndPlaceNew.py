import gymnasium as gym
import time
import panda_gym

env = gym.make("PandaPickAndPlaceNew-v3",render_mode="human",interest_object="object_1",target_object="object_2")

observation, info = env.reset()

for _ in range(50000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)
    print("observation ", observation )
    if terminated or truncated:
        observation, info = env.reset()

env.close()
