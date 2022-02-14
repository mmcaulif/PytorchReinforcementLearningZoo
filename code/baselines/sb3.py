import gym
import numpy as np

from stable_baselines3 import TD3

env_name = "Pendulum-v1"
#env_name = "BipedalWalker-v3"
#env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)

model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=10)

"""
-TD3 should get LunarLanderContinuous-v2 in ~550 episodes, or at least reach a positive value around ~300 episodes
"""