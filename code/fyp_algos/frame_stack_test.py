import gym
from gym.wrappers import FrameStack
import numpy as np
import torch
from code.utils.attacker import Attacker

stacks = 2
env = gym.make('CartPole-v0')
env = Attacker(env)
env = FrameStack(env, 2)
s_t = env.reset()
for i in range(20):
    a_t = env.action_space.sample()
    s_tp1, r_t, d, i_t = env.step(a_t)
    #print(s_tp1[:stacks], r_t)
    s_tp1 = np.concatenate([s_tp1[0], s_tp1[1]])
    print(i, s_tp1, r_t, i_t, '\n')
    s_t = s_tp1
    if d:
        s_t = env.reset()
