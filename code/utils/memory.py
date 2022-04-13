import torch
import numpy as np

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
        states = torch.as_tensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        policies = torch.stack(self.policies).float()
        dones = torch.IntTensor(self.dones)
        qty = self.qty
        
        self.states, self.actions, self.rewards, self.policies, self.dones = [], [], [], [], []
        self.qty = 0
        
        return states, actions, rewards, policies, dones, qty