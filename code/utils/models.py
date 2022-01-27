import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Q_val(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_val, self).__init__()
		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, action_dim)

	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_val, self).__init__()
		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 64)
		self.l4 = nn.Linear(64, action_dim)

	def forward(self, state):
		q = F.relu(self.l1(state))
		adv = F.relu(self.l2(q))
		val = F.relu(self.l3(q))
		q = self.l4(val + adv)
		return q