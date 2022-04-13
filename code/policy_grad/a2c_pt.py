import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

#https://colab.research.google.com/github/yfletberliac/rlss-2019/blob/master/labs/DRL.01.REINFORCE%2BA2C.ipynb#scrollTo=xDifFS9I4X7A
#https://omegastick.github.io/2018/06/25/easy-a2c.html
#https://github.com/floodsung/a2c_cartpole_pytorch  <- very good

class Model(torch.nn.Module):
    def __init__(self, input_size, action_size):
        super(Model, self).__init__()
        
        self.critic = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, 1)
        )
        self.actor = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, s):
        v = self.critic(s)
        pi = self.actor(s)
        return v, pi

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.rewards, self.policies = [], [], [], []
        self.qty = 0
    
    def push(self, state, action, reward, policy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.qty += 1
    
    def pop_all(self):
        states = torch.as_tensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1)
        policies = torch.stack(self.policies).float()
        
        self.states, self.actions, self.rewards, self.policies = [], [], [], []
        self.qty = 0
        
        return states, actions, rewards.squeeze(-1), policies

    
def select_action(x):
    x_t = int(torch.multinomial(x, 1).detach())
    x_dist = Categorical(x)
    return x_t, x_dist.log_prob(torch.tensor(x_t)), x_dist.entropy().mean()

env_name = "CartPole-v1"
#env_name = "LunarLander-v2"
env = RecordEpisodeStatistics(gym.make(env_name))
s_t = env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

a2c_model = Model(obs_dim, act_dim)
optimizer = torch.optim.Adam(a2c_model.parameters(), lr=0.0007, eps=1e-5)   #ppo optimiser

buffer = Memory()

def calc_returns(r_rollout, final_r):
    discounted_r = torch.zeros_like(r_rollout)
    for t in reversed(range(0, len(r_rollout))):
        final_r = final_r *  0.99 + r_rollout[t]
        discounted_r[t] = final_r
    return discounted_r

avg_r = deque(maxlen=200)
count = 0

for i in range(50000):

    r_trajectory = 0

    while buffer.qty <= 5:
        a_pi = a2c_model(torch.from_numpy(s_t).float())[1]
        a_t = int(torch.multinomial(a_pi, 1).detach())
        s_tp1, r, d, info = env.step(a_t)
        buffer.push(s_t, a_t, r, a_pi)
        s_t = s_tp1
        if d:
            s_t = env.reset()
            count += 1
            avg_r.append(int(info["episode"]["r"]))
            if count % 20 == 0:
                print(f'Episode: {count} | Average reward: {sum(avg_r)/len(avg_r)} | Rollouts: {i} | [{len(avg_r)}]')
            break

    if not d:
        r_trajectory, _ = a2c_model(torch.from_numpy(s_tp1).float())  
    
    s_rollout, a_rollout, r_rollout, pi_rollout = buffer.pop_all()
    
    Q = calc_returns(r_rollout, r_trajectory)
        
    V = a2c_model(s_rollout.float())[0].squeeze(-1)
    
    critic_loss = F.mse_loss(Q, V)
    
    dist_rollout = Categorical(pi_rollout)   

    entropy = dist_rollout.entropy().mean() #ppo entropy loss
    adv = Q - V.detach()

    log_probs = dist_rollout.log_prob(a_rollout)
    actor_loss = -(log_probs * adv.detach()).mean()
    loss = actor_loss + (critic_loss * 0.5) - (entropy * 0.01)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(a2c_model.parameters(), 0.5) #ppo gradient clipping
    optimizer.step()

    

