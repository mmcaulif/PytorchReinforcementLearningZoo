import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import gym
import copy
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
import numpy as np
import time
from pettingzoo.mpe import simple_v2

from code.utils.models import Q_val
# from code.utils.memory import

from code.utils.memory import ReplayBuffer


class GumbelDDPG():
	def __init__(
		self,
		environment,
		network,
		pi,
		gamma=0.99,
		train_after=50000,
		train_freq=4,
		target_update=1000,
		batch_size=64,
		verbose=500,
		learning_rate=1e-4,
		max_grad_norm=10,
		tau=5e-3
	):
		self.environment = environment
		self.pi = pi
		self.q_func = network
		self.q_target = copy.deepcopy(self.q_func)
		self.gamma = gamma
		self.train_after = train_after
		self.train_freq = train_freq
		self.target_update = target_update
		self.batch_size = batch_size
		self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
		self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=learning_rate)
		self.max_grad_norm = max_grad_norm
		self.tau = tau

		self.EPS_END = 0.05
		self.EPS = 0.9
		self.EPS_DECAY = 0.999

		self.verbose = verbose

	def update(self, batch):
		s, a, r, s_p, d = batch

		s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
		a = torch.from_numpy(np.array(batch.a)).type(torch.float32)
		r = torch.FloatTensor(batch.r).unsqueeze(1)
		s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
		d = torch.IntTensor(batch.d).unsqueeze(1)

		q = self.q_func(s, a)

		# q_p = self.q_target(s_p, a_p)

		# print(q.shape, q_p.shape)

		with torch.no_grad():
			a_p = self.pi(s_p)
			q_p = self.q_target(s_p, a_p)
			y = r + self.gamma * q_p * (1 - d)

		loss = F.mse_loss(q, y)

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
		self.optimizer.step()

		# policy training
		a_sampled, pol_out = self.pi(s, hard=False)
		policy_loss = -self.q_func(s, a_sampled).mean()
		policy_loss += (pol_out**2).mean() * 1e-3    # This line might be crucial, unsure why, can still learn without it tho

		self.pi_optimizer.zero_grad()
		policy_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
		self.pi_optimizer.step()

		return loss

	def hard_update(self):
		self.q_target = copy.deepcopy(self.q_func)

	def soft_update(self):
		for target_param, param in zip(self.q_target.parameters(), self.q_func.parameters()):
				target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

	def select_action(self, s):

		self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
		if torch.rand(1) > self.EPS:
			one_hot = self.pi(torch.from_numpy(s).float()).detach()
			a = torch.argmax(one_hot).item()

		else:
			one_hot = torch.zeros(self.environment.action_space.n)
			a = self.environment.action_space.sample()
			one_hot[a] = 1

		"""logits = self.pi(torch.from_numpy(s).float(), hard=False)[0].detach()

		one_hot = torch.zeros(len(logits))
		a = torch.distributions.Categorical(logits).sample()
		one_hot[a] = 1"""

		return a, one_hot.numpy()    # one_hot.numpy()

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()
		self.l1 = nn.Linear(state_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, action_dim)

	def forward(self, state, hard=True):
		pol_out = F.relu(self.l1(state))
		pol_out = F.relu(self.l2(pol_out))
		pol_out = self.l3(pol_out)

		# Gumbel-softmax
		gumbels = torch.distributions.Uniform(0,1).sample(pol_out.shape)
		gumbels = -torch.log(-torch.log(gumbels+1e-20)+1e-20)

		logits = F.softmax(pol_out + gumbels, dim=-1)

		if hard:
			max_actions = logits.argmax(dim=-1).unsqueeze(-1)
			logits_hard = torch.zeros(logits.shape).scatter(-1, max_actions, 1)
			return logits_hard - logits.detach() + logits

		return logits, pol_out

class ddpg_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ddpg_Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, 1)

	def forward(self, state, action):
		try:
			sa = torch.cat([state, action], 1)
		except:
			sa = torch.cat([state, action], -1)

		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

class SingleAgent():
	def __init__(self, env):
		self.env = env
		# self.agents = self.env.agents
		self.observation_space = env.observation_space('agent_0')
		self.action_space = env.action_space('agent_0')

	def step(self, a):
		a = {'agent_0': a}
		s, r, d, t, i = self.env.step(a)

		return s['agent_0'], r['agent_0'], d['agent_0'], t['agent_0'], i

	def reset(self):
		return self.env.reset()['agent_0']

def main():
	env = simple_v2.parallel_env(max_cycles=50)
	print(env.possible_agents)
	env = SingleAgent(env)

	obs_dim = env.observation_space.shape[0]
	act_dim = 5	# env.action_space.shape	# [0]

	replay_buffer = ReplayBuffer(buffer_len=50000)

	net = ddpg_Critic(obs_dim, act_dim)

	policy_net = Policy(obs_dim, act_dim)

	dqn_agent = GumbelDDPG(
		 env,
		 net,
		 policy_net,
		 gamma=0.99,
		 train_after=10000,
		 target_update=200,
		 batch_size=32,
		 verbose=2000,
		 learning_rate=0.0005)

	episodes = 0
	s_t = env.reset()

	episodic_rewards = deque(maxlen=50)

	eps_rew = 0

	for i in range(200000):
		a_t, a_oh = dqn_agent.select_action(s_t)

		# print(i)
		s_tp1, r_t, done, trun, info = env.step(a_t)
		eps_rew += r_t
		replay_buffer.append(s_t, a_oh, r_t, s_tp1, done)
		s_t = s_tp1

		if len(replay_buffer) >= dqn_agent.batch_size and i >= dqn_agent.train_after:

			if i % dqn_agent.train_freq == 0:
				batch = replay_buffer.sample(dqn_agent.batch_size)
				loss = dqn_agent.update(batch)

			if i % dqn_agent.target_update == 0:
				# dqn_agent.hard_update()
				dqn_agent.soft_update()

			if i % dqn_agent.verbose == 0:
				avg_r = sum(episodic_rewards) / len(episodic_rewards)
				print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

		# print(done)
		if done or trun:
			episodes += 1
			episodic_rewards.append(eps_rew)
			eps_rew = 0
			s_t = env.reset()



	# Render Trained agent

	env = simple_v2.parallel_env(render_mode='human', max_cycles=50)
	print(env.possible_agents)
	env = SingleAgent(env)
	eps_rew = 0
	s_t = env.reset()
	while True:
		# env.render()
		a_t, _ = dqn_agent.select_action(s_t)
		s_tp1, r_t, done, trun, info = env.step(a_t)
		eps_rew += r_t
		s_t = s_tp1
		time.sleep(0.1)
		if done or trun:
			print(f'Episode Complete, reward = {eps_rew}')
			eps_rew = 0
			s_t = env.reset()

if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()

