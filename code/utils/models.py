from email import policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class td3_Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(td3_Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = torch.tanh(self.l3(a))
		return torch.mul(a, self.max_action)

class td3_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(td3_Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


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

class ddpg_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ddpg_Critic, self).__init__()

		self.critic = torch.nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
			nn.ReLU(),
            nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 1)
        )

	def forward(self, state, action):
		try:
			sa = torch.cat([state, action], 1)
		except:	
			sa = torch.cat([state, action], -1)

		q = self.critic(sa)
		return q, q

	def q1_forward(self, state, action):
		try:
			sa = torch.cat([state, action], 1)
		except:	
			sa = torch.cat([state, action], -1)

		q = self.critic(sa)
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
		v = self.val(q)
		a = self.adv(q)
		return v + (a - a.mean())

	def get_rel_adv(self, state):
		q = F.relu(self.l1(state))
		a = self.adv(q)
		return (a - a.mean())

class PPO_model(torch.nn.Module):
	def __init__(self, input_size, action_size, net_size=256):
		super(PPO_model, self).__init__()
        
		self.critic = torch.nn.Sequential(
			nn.Linear(input_size, net_size),
			nn.Tanh(),
			nn.Linear(net_size, 1))

		self.actor = torch.nn.Sequential(
			nn.Linear(input_size, net_size),
			nn.Tanh(),
			nn.Linear(net_size, action_size),
			nn.Softmax(dim=-1))

	def forward(self, s):
		v = self.critic(s)
		pi = self.actor(s)
		return v, pi

	def get_dist(self, s):
		policy = self.actor(s)
		dist = Categorical(policy)
		return dist

class PPO_cont_model(torch.nn.Module):
	def __init__(self, input_size, action_size, net_size=256):
		super(PPO_cont_model, self).__init__()

		self.critic = torch.nn.Sequential(
			nn.Linear(input_size, net_size),
			nn.Tanh(),
			nn.Linear(net_size, 1))

		self.actor = torch.nn.Sequential(
			nn.Linear(input_size, net_size),
			nn.Tanh(),
			nn.Linear(net_size, action_size))

		self.log_std = nn.Parameter(torch.ones(action_size))

	def forward(self, s):
		v = self.critic(s)
		mu = self.actor(s)
		return v, mu

	def get_dist(self, s):
		mu = self.actor(s)
		std = torch.exp(self.log_std.expand_as(mu))
		dist = Normal(mu, std)
		return dist

class sac_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(sac_Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)

    def forward(self, state):
        state = self.fc(state)
        mean = self.fc_mean(state)
        log_std = self.fc_std(state)
        log_std = torch.clamp(log_std, -5, 2)
        return mean, log_std
