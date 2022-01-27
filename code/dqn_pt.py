import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import random
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

from utils.models import Q_val

class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

def update(batch):
    s = torch.from_numpy(np.array(batch.s))
    a = torch.IntTensor(batch.a).unsqueeze(1)    #remove for LunarLanderContinuous-v2
    r = torch.FloatTensor(batch.r).unsqueeze(1)
    s_p = torch.from_numpy(np.array(batch.s_p))
    d = torch.IntTensor(batch.d).unsqueeze(1)

    q = torch.gather(q_func(s), 1, a.long())

    y = r + 0.99 * torch.max(q_target(s_p).detach()) * (1 - d)

    loss = F.mse_loss(q, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

#env_name = 'LunarLanderContinuous-v2'
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = RecordEpisodeStatistics(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

replay_buffer = deque(maxlen=100000)

q_func = Q_val(obs_dim, act_dim)
q_target = copy.deepcopy(q_func)
optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-3)

s_t = env.reset()

losses = []
episodic_rewards = []

for i in range(150000):
    a_t = torch.argmax(q_func(torch.from_numpy(s_t)).detach())
    
    s_tp1, r_t, done, info = env.step(int(a_t))
    
    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if len(replay_buffer) >= 128:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        loss = update(batch)
        losses.append(loss)
        avg_losses = (sum(losses[-100:]))/100
        avg_r = (sum(episodic_rewards[-100:]))/100

        if i % 100 == 0: print(f"Timestep: {i} | Avg. Loss: {avg_losses} | Avg. Reward: {avg_r}")  

        if i % 1000 == 0: q_target = copy.deepcopy(q_func) 

    if done:
        episodic_rewards.append(int(info['episode']['r']))
        s_tp1 = env.reset()

    s_t = s_tp1

#Render Trained agent
s_t = env.reset()
while True:
    env.render()
    a_t = torch.argmax(q_func(torch.from_numpy(s_t)).detach())
    s_tp1, r_t, done, _ = env.step(a_t)
    if done:
        s_tp1 = env.reset()

    s_t = s_tp1