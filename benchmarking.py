import gym
import numpy as np

from stable_baselines3 import PPO, TD3, DQN, A2C

#env_name = "Pendulum-v1"
#env_name = "BipedalWalker-v3"
#env_name = "LunarLanderContinuous-v2"
#env_name = "LunarLander-v2"
#env_name = "CartPole-v1"

env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
env = gym.make(env_name)

#model = TD3("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=1000000, log_interval=10)

#model = DQN("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=1000000, log_interval=25)

model = PPO("MlpPolicy", env, verbose=20)
model.learn(total_timesteps=1000000)

"""
-DQN should get LunarLander-v2 in ~X episodes, or at least reach a positive value around ~950 episodes
-TD3 should get LunarLanderContinuous-v2 in ~550 episodes, or at least reach a positive value around ~300 episodes
"""