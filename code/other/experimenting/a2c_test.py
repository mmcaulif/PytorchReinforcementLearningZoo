from audioop import avg
from collections import deque
from statistics import mean
import time
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

class Actor(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dims, 256),
            nn.Tanh(),
            nn.Linear(256, act_dims),
            nn.Softmax(dim=0)
        )

    def forward(self, s):
        return self.network(s)

class Critic(nn.Module):
    def __init__(self, obs_dims):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dims, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, s):
        return self.network(s)

class A2C():
    def __init__(self, policy, critic, n_steps=5, gamma=0.99, lr=0.0007):
        self.policy = policy
        self.critic = critic
        self.n_steps = n_steps
        self.gamma = gamma
        self.lr = lr
        self.optim = torch.optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()), lr=self.lr, eps=1e-5)
        lambda1 = lambda epoch: 0.99 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda1)

    def act(self, s):
        dist = self.policy(torch.tensor(s).float())
        a = Categorical(dist).sample().numpy()
        return a

    def calc_disc_returns(self, returns, last_r):
        discounted_returns = []
        total_returns = last_r
        for i, r in enumerate(reversed(returns)):
            total_returns += r * (self.gamma ** i)
            discounted_returns.append(total_returns)

        discounted_returns.reverse()
        return torch.tensor(discounted_returns)

    def update(self, states, actions, rewards, done, last_state):
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions)

        if not done:
            last_r = self.critic(torch.tensor(last_state).float())
        else:
            last_r = 0

        returns = self.calc_disc_returns(rewards, last_r).float()

        values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)

        advantages = returns - values.detach()

        dist_pi = self.policy(states)
        logits = Categorical(dist_pi).log_prob(actions)
        policy_loss = -(logits * advantages.detach()).mean()

        loss = policy_loss + (0.5 * critic_loss)

        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(list(self.policy.parameters()) + list(self.critic.parameters()), 0.5)
        self.optim.step()
        # self.scheduler.step()
        return loss

env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
env = gym.make(env_name)
timesteps = 500000

agent = A2C(
    policy=Actor(env.observation_space.shape[0], env.action_space.n),
    critic = Critic(env.observation_space.shape[0]),)

s_t = env.reset()
r_sum = 0
avg_r = deque(maxlen=20)

r_traj = []
a_traj = []
s_traj = []

i = 0
count = 0

while True:
    i += 1
    s_traj.append(s_t)
    a_t = agent.act(s_t)
    a_traj.append(int(a_t))
    s_tp1, r_t, d, _ = env.step(a_t)

    r_sum += r_t
    r_traj.append(r_t)

    if d or len(r_traj) == agent.n_steps:
        agent.update(s_traj, a_traj, r_traj, d, s_tp1)
        r_traj = []
        a_traj = []
        s_traj = []

    if d:
        # agent.scheduler.step()
        avg_r.append(r_sum)

        if mean(avg_r) >= 225:
            count += 1
            if count == 3:
                break

        else:
            count = 0

        r_sum = 0
        s_t = env.reset()

    if i % 1000 == 0 and len(avg_r) > 0:
        print(f'{i}: {np.round(mean(avg_r))}, [{len(avg_r)}]')
    
    s_t = s_tp1

r_sum = 0
s_t = env.reset()
while True:
    env.render()
    time.sleep(0.02)
    a_t = agent.act(s_t)
    s_tp1, r_t, d, _ = env.step(a_t)
    s_t = s_tp1
    r_sum += r_t
    if d:
        print(r_sum)
        r_sum = 0
        s_t = env.reset()
    

    