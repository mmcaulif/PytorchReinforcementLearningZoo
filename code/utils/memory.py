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
        self.states, self.actions, self.rewards, self.dones, self.policies, self.next_states = [], [], [], [], [], []
        self.v_hiddens, self.v_hnexts, self.pi_hiddens = [], [], []
    
    def push(self, state, action, reward, next_state, done, policy, v_hidden, v_hnext, pi_hidden):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.policies.append(policy)
        self.v_hiddens.append(v_hidden)
        self.v_hnexts.append(v_hnext)
        self.pi_hiddens.append(pi_hidden)

    def pop_all(self):
        states = torch.from_numpy(np.array(self.states)).float()
        actions = torch.from_numpy(np.array(self.actions)).float()
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.from_numpy(np.array(self.next_states)).float()
        dones = torch.IntTensor(self.dones)
        policies = torch.FloatTensor(self.policies)
        v_hiddens = torch.stack(self.v_hiddens)
        v_hnexts = torch.stack(self.v_hnexts)
        pi_hiddens = torch.stack(self.pi_hiddens)
        
        self.states, self.actions, self.rewards, self.dones, self.policies, self.next_states = [], [], [], [], [], []
        self.v_hiddens, self.v_hnexts, self.pi_hiddens = [], [], []
        
        return states, actions, rewards, next_states, dones, policies, v_hiddens, v_hnexts, pi_hiddens

    def __len__(self):
        return len(self.states)
        