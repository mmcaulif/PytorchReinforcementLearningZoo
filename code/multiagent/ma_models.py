import sys
import torch
import torch.nn as nn


# MADDPG models

HIDDEN_DIMS = 128

class central_Critic(nn.Module):
    def __init__(self,  sum_state_dim, sum_action_dim):
        super(central_Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(sum_state_dim + sum_action_dim, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, 1)
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, states, actions):
        sa = torch.cat(list(states.values()) + list(actions.values()), -1)        
        q = self.net(sa)
        return q.squeeze(-1)

class central_twinq_Critic(nn.Module):
    def __init__(self,  sum_state_dim, sum_action_dim):
        super(central_twinq_Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(sum_state_dim + sum_action_dim, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, 1)
        ).apply(self.init)

        self.net2 = nn.Sequential(
            nn.Linear(sum_state_dim + sum_action_dim, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, 1)
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, states, actions):
        sa = torch.cat(list(states.values()) + list(actions.values()), -1)        
        q1, q2 = self.net1(sa), self.net2(sa)
        return q1.squeeze(-1), q2.squeeze(-1)

    def q1(self, states, actions):
        sa = torch.cat(list(states.values()) + list(actions.values()), -1)        
        q1 = self.net1(sa)
        return q1.squeeze(-1)

class local_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(local_Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS, action_dim),
            nn.Tanh()
        ).apply(self.init)
        self.max_action = max_action

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, states):        
        a = self.net(states)
        # return torch.mul(a, self.max_action)
        return (torch.mul(a, self.max_action)+1)/2