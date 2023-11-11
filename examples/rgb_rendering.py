import gymnasium as gym
import cv2

# from numpngw import write_apng  # pip install numpngw

import panda_gym

env = gym.make("PandaStack-v3", render_mode="rgb_array")
# images = []

# initialize cv2.window
cv2.namedWindow("Scene")
observation, info = env.reset()
# images.append(env.render())
cv2.imshow("Scene", env.render())

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # images.append(env.render())
    cv2.imshow("Scene", env.render())

    if terminated or truncated:
        observation, info = env.reset()
        # images.append(env.render())
        cv2.imshow("Scene", env.render())

    if cv2.waitKey(1) & 0xFF == ord("q"):  # wait for 1 millisecond
        break


env.close()

# write_apng("stack.png", images, delay=40)
