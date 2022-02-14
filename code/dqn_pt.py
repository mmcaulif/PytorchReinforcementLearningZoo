from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import random
import math
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

from utils.models import Q_val, Q_duelling

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int

class DQN():
    def __init__(self, gamma, train_after, target_update, network):
        self.gamma = gamma
        self.train_after = train_after
        self.target_update = target_update
        self.q_func = network
        self.q_target = copy.deepcopy(self.q_func)
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=1e-3)

        self.EPS_END = 0.05
        self.EPS_START = 0.9
        self.EPS_DECAY = 200

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.IntTensor(batch.a).unsqueeze(1)    #remove for LunarLanderContinuous-v2
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)

        q = torch.gather(self.q_func(s), 1, a.long())

        y = r + 0.99 * torch.max(self.q_target(s_p).detach()) * (1 - d)

        loss = F.mse_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        self.q_target = copy.deepcopy(self.q_func) 

    def select_action(self, s, i):

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i / self.EPS_DECAY)

        if torch.rand(1) > eps_threshold:
            a = torch.argmax(self.q_func(torch.from_numpy(s)).detach())
        else:
            a = env.action_space.sample()

        return a


#env_name = 'LunarLanderContinuous-v2'
env_name = 'CartPole-v1'
env = gym.make(env_name)
env = RecordEpisodeStatistics(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

replay_buffer = deque(maxlen=100000)

model = Q_val(obs_dim, act_dim)
#model = Q_duelling(obs_dim, act_dim)
dqn_agent = DQN(0.99, 50000, 1000, model)

#optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-3)

s_t = env.reset()

losses = []
episodic_rewards = []

for i in range(150000):
    a_t = dqn_agent.select_action(s_t, i)
    s_tp1, r_t, done, info = env.step(int(a_t))
    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if len(replay_buffer) >= 128 and i > dqn_agent.train_after:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        loss = dqn_agent.update(batch)
        losses.append(loss)
        avg_losses = (sum(losses[-100:]))/100
        avg_r = (sum(episodic_rewards[-100:]))/100

        if i % 100 == 0: print(f"Timestep: {i} | Avg. Loss: {avg_losses} | Avg. Reward: {avg_r}")  
        if i % dqn_agent.target_update == 0: dqn_agent.update_target()

    if done:
        episodic_rewards.append(int(info['episode']['r']))

        s_tp1 = env.reset()

    s_t = s_tp1

#Render Trained agent
s_t = env.reset()
while True:
    env.render()
    a_t = dqn_agent.select_action(s_t)
    s_tp1, r_t, done, _ = env.step(a_t)
    if done:
        s_tp1 = env.reset()

    s_t = s_tp1