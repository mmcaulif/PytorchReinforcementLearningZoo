import random
import torch
import numpy as np
from typing import NamedTuple
from collections import deque

class Rollout_Memory(object):

    def __init__(self):
        self.states, self.actions, self.rewards, self.policies, self.dones = [], [], [], [], []
        self.qty = 0
    
    def push(self, state, action, reward, policy, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.dones.append(done)
        self.qty += 1
    
    def pop_all(self):
        states = torch.as_tensor(np.array(self.states)).float()
        actions = torch.as_tensor(np.array(self.actions)).float()
        rewards = torch.FloatTensor(self.rewards)
        policies = torch.stack(self.policies).float()
        dones = torch.IntTensor(self.dones)
        qty = self.qty
        
        self.states, self.actions, self.rewards, self.policies, self.dones = [], [], [], [], []
        self.qty = 0
        
        return states, actions, rewards, policies, dones, qty

class Aux_Memory(object):

    def __init__(self):
        self.states, self.target_val, self.old_val = [], [], []
        self.qty = 0
    
    def push(self, state, target_val, old_val):
        self.states.append(state)
        self.target_val.append(target_val)
        self.old_val.append(old_val)
        self.qty += 1
    
    def pop_all(self):
        states = self.states[0]
        target_val = self.target_val[0]
        old_val = self.old_val[0]
        qty = self.qty
        
        self.states, self.target_val, self.old_val = [], [], []
        self.qty = 0
        
        return states, target_val, old_val, qty

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
        