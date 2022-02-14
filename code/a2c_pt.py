from optparse import Values
from tkinter import N
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

#https://colab.research.google.com/github/yfletberliac/rlss-2019/blob/master/labs/DRL.01.REINFORCE%2BA2C.ipynb#scrollTo=xDifFS9I4X7A
#https://omegastick.github.io/2018/06/25/easy-a2c.html

class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out
    
    def select_action(self, x):
        x = self(torch.tensor(x))
        return int(torch.multinomial(x, 1).detach().numpy()), x

class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
    
    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(true_value)
    
    def pop_all(self):
        #states = torch.stack(self.states)
        states = torch.as_tensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1)
        
        self.states, self.actions, self.rewards = [], [], []
        
        return states, actions, rewards.squeeze(-1)

def discount_reward(r, gamma,final_r):
    #discounted_r = np.zeros_like(r)
    discounted_r = torch.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.squeeze(-1)


env = gym.make("CartPole-v1")
s_t = env.reset()

policy = Actor(4, 2)
value = Critic(4)
#optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
buffer = Memory()

for i in range(1, 100):
    env.render()
    a_t, a_dist = policy.select_action(s_t)
    s_tp1, r, d, _ = env.step(a_t)
    buffer.push(s_t, a_t, r)

    a_dist = torch.distributions.Categorical(a_dist)
    log_probs = a_dist.log_prob(torch.tensor(a_t))
    entropy = a_dist.entropy().mean()

    if len(buffer.rewards) >= 8:
        states, actions, rewards = buffer.pop_all()
        #final_r = value(states[-1])
        final_r = 1
        true_values = discount_reward(rewards, 0.99, final_r)

        print(f"timestep: {i}, Final R: {final_r}")
        print(rewards)
        print(true_values)
        values = value(states).detach().squeeze(-1)
        print(values)
        advantages = true_values - values
        print(advantages)
        critic_loss = F.mse_loss(true_values, values)
        print(critic_loss, '\n')

        """critic_loss = F.mse_loss(true_values, values)
        advantages = true_values - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        total_loss = (0.5 * critic_loss) + actor_loss - (0.01 * entropy)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()"""

    if d:
        s_tp1 = env.reset()

    s_t = s_tp1

