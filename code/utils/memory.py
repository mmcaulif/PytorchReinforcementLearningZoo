import random
import sys
import torch
import numpy as np
from typing import NamedTuple
from collections import deque


class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

class ReplayBuffer():
    def __init__(self, buffer_len):
        self.buffer = deque(maxlen=buffer_len)

    def append(self, s, a, r, s_p, d):
        self.buffer.append([s, a, r, s_p, d])

    def sample(self, k):
        return Transition(*zip(*random.sample(self.buffer, k)))

    def __len__(self):
        return len(self.buffer)

class Rollout_Memory(object):
    def __init__(self):
        self.states, self.actions, self.rewards, self.dones, self.policies = [], [], [], [], []
    
    def push(self, state, action, reward, done, policy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.policies.append(policy)
    
    def pop_all(self):
        states = torch.from_numpy(np.array(self.states)).float()
        actions = torch.from_numpy(np.array(self.actions)).float()
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.IntTensor(self.dones)
        policies = torch.FloatTensor(self.policies)
        
        self.states, self.actions, self.rewards, self.dones, self.policies = [], [], [], [], []
        
        return states, actions, rewards, dones, policies

    def __len__(self):
        return len(self.states)
        