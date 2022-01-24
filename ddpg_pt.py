import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
import copy
from collections import deque
from typing import NamedTuple

"""
implementations for help:
https://github.com/sfujim/TD3/blob/master/TD3.py
https://colab.research.google.com/drive/19-z6Go3RydP9_uhICh0iHFgOXyKXgX7f#scrollTo=zuw702kdRr4E
"""



class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

def noisy_action(a_t, env):
    noise = (torch.randn_like(a_t) * 0.2).clamp(-0.5, 0.5)
    return (a_t + noise).clamp(env.action_space.low[0],env.action_space.high[0])

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return torch.tanh(self.l3(a)) * self.max_action

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
replay_buffer = Memory(500)

actor = Actor(2, 1, env.action_space.high[0])

critic = Critic(2, 1)
critic_target = copy.deepcopy(critic)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

s_t = env.reset()

for i in range(101):
    #env.render()
    a_t = torch.from_numpy(env.action_space.sample())
    a_t = noisy_action(a_t, env)

    s_tp1, r_t, d, _ = env.step(a_t)

    replay_buffer.push(s_t, a_t, r_t, s_tp1, d)

    if len(replay_buffer) > 100:
        s_t, a_t, r_t, s_tp1, done = replay_buffer.sample(1)

        s_t = torch.FloatTensor(s_t)
        a_t = torch.FloatTensor(a_t).unsqueeze(1)
        r_t = torch.FloatTensor(r_t)
        s_tp1 = torch.FloatTensor(s_tp1)

        print("T: ", s_t, a_t)
        q = critic(s_t, a_t)

        a_tp1 = actor(s_tp1)
        print("T + 1: ", s_tp1, a_tp1)

        target_q = critic_target(s_tp1, a_tp1).detach()
        #print(f"{target_q}")

        #y = r_t + (1 - done) * 0.99 * target_q
        #print(f"{y}")
        
        critic_loss = F.mse_loss(q, target_q)
        print(f"Loss: {critic_loss}")

    if d:
        s_tp1 = env.reset()

    s_t = s_tp1

env.close()