# WARNING, This file will not be functional until stable-baselines3 is compatible
# with gymnasium. See https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.
import gymnasium as gym
from stable_baselines3 import PPO, HerReplayBuffer

import panda_gym

env = gym.make("PandaPickAndPlace-v3",reward_type="sparse") #interest_object="object_0",target_object="object_1")

model = PPO("MultiInputPolicy", env, verbose=1,tensorboard_log="./ppo_pickplace_ori_tensorboard/")

model.learn(total_timesteps=30000)

model.save("ppo_pandapickplace")

vec_env = model.get_env()

del model

model = PPO.load("ppo_pandapickplace")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)

    observation, reward, terminated, info = vec_env.step(action)
    
    vec_env.render("human")
    
    if terminated :
        print("observation, reward, terminated, info ", observation, reward, terminated,  info )
        obs = vec_env.reset()


