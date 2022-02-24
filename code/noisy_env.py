from collections import deque
from distutils.log import info
import random
from typing import NamedTuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym import Wrapper
from dqn_pt import DQN
from utils.models import Q_val

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int
    t: int

replay_buffer = deque(maxlen=100000)

#network
class CIQ_Net(nn.Module):
	def __init__(self, state_dim):
		super(CIQ_Net, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(state_dim, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 32),
                                     nn.ReLU())

	def ciq_forward(self, state):
		Z = self.encoder(state)

		return Z

#agent, subclass of DQN
class CIQ(DQN):    #test that this functions correctly as a vanilla dqn
    def __init__(self, tau=1e-3):
        self.tau = tau

    def soft_update_target(self):
        for self.q_target, self.q_func in zip(self.q_target.parameters(), self.q_func.parameters()):
            self.q_target.data.copy_(self.tau*self.q_func.data + (1.0-self.tau)*self.q_target.data)

    def ciq_update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.IntTensor(batch.a).unsqueeze(1)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_training = torch.IntTensor(batch.t)
        loss = 0
        print(i_training)

class GaussianNoise(Wrapper):
    def __init__(self, env, p=0.5, var=1):
        super().__init__(env)
        self.env = env
        self.p = p
        self.var = var
        self.obs_low = env.observation_space.low[0]
        self.obs_high = env.observation_space.high[0]

    def step(self, action):  
        next_state, reward, done_bool, _ = super().step(action)

        noise = torch.normal(mean=torch.zeros_like(torch.from_numpy(next_state)), std=self.var).numpy()
        
        I = 0
        if torch.rand(1) < self.p: I = 1
        
        gaussian_state = np.clip((next_state + noise), self.obs_low, self.obs_high)
        next_state = I * gaussian_state + (1 - I) * next_state

        return next_state, reward, done_bool, I


env = gym.make('CartPole-v1')
env = GaussianNoise(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
ciq_agent = CIQ(env, 0.99, 50000, 1000, 64, Q_val(obs_dim, act_dim), 500)
s_t = env.reset()

for i in range(20):
    a_t = ciq_agent.select_action(s_t)
    
    s_tp1, r_t, d, i_t = env.step(a_t)

    #print(s_tp1, i_t)

    replay_buffer.append([s_t, a_t, r_t, s_tp1, d, i_t])
    
    s_t = s_tp1

    if d:
        s_t = env.reset()
