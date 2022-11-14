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

class Actor(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dims, 256),
            nn.Tanh(),
            nn.Linear(256, act_dims)
        )

    def forward(self, s):
        return F.softmax(self.network(s), dim=0)

class Reinforce():
    def __init__(self, policy, gamma=0.99, lr=3e-4):
        self.policy = policy
        self.gamma = gamma
        self.kr = lr
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        pass

    def act(self, s):
        dist = self.policy(torch.tensor(s).float())
        a = Categorical(dist).sample().numpy()
        return a

    def calc_disc_returns(self, returns):
        discounted_returns = []
        total_returns = 0
        for i, r in enumerate(reversed(returns)):
            total_returns += r * (self.gamma ** i)
            discounted_returns.append(total_returns)

        discounted_returns.reverse()
        return discounted_returns

    def update(self, states, actions, returns):
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions)
        returns = torch.Tensor(self.calc_disc_returns(returns))        

        dist_pi = self.policy(states)
        logits = Categorical(dist_pi).log_prob(actions)
        # print(f'{returns}\n{logits}')
        loss = -(logits.sum() * returns).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss


env = gym.make('CartPole-v1')
timesteps = 100000

agent = Reinforce(policy=Actor(env.observation_space.shape[0], env.action_space.n))

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

    if d:
        avg_r.append(r_sum)

        if mean(avg_r) >= 225:
            count += 1
            if count == 3:
                break

        else:
            count = 0

        r_sum = 0
        s_t = env.reset()

        agent.update(s_traj, a_traj, r_traj)
        r_traj = []
        a_traj = []
        s_traj = []

    if i % 1000 == 0 and len(avg_r) > 0:
        print(f'{i}: {mean(avg_r), [{len(avg_r)}]}')
    
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
    

    