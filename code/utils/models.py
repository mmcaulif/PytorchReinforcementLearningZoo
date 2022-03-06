import torch
import torch.nn as nn
import torch.nn.functional as F

class td3_Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(td3_Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = torch.tanh(self.l3(a))
		return torch.mul(a, self.max_action)

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

class td3_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(td3_Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		try:
			sa = torch.cat([state, action], 1)
		except:	
			sa = torch.cat([state, action], -1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1_forward(self, state, action):
		try:
			sa = torch.cat([state, action], 1)
		except:	
			sa = torch.cat([state, action], -1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

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
		super(Q_duelling, self).__init__()
		self.l1 = nn.Linear(state_dim, 64)

		self.adv = nn.Sequential(
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim)
		)

		self.val = nn.Sequential(
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

	def forward(self, state):
		q = F.relu(self.l1(state))
		return self.val(q) + self.adv(q)