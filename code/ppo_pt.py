import copy
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

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py  shows off the ratios well

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
        qty = self.qty
        
        self.states, self.actions, self.rewards, self.policies = [], [], [], []
        self.qty = 0
        
        return states, actions, rewards.squeeze(-1), policies, qty

    
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

ppo_model = Model(obs_dim, act_dim)
optimizer = torch.optim.Adam(ppo_model.parameters(), lr=2.5e-4, eps=1e-5)   #ppo optimiser

buffer = Memory()

def calc_returns(r_rollout, final_r):
    discounted_r = torch.zeros_like(r_rollout)
    gae_tm1 = 0
    for t in reversed(range(0, len(r_rollout))):
        final_r = final_r *  0.99 + r_rollout[t]
        #discounted_r[t] = gae_tm1 = final_r * 0.99 * 0.95 * gae_tm1    #attempt at ppo
        discounted_r[t] = final_r
    return discounted_r

def calc_gae(r_rollout, s_rollout, final_r):
    adv_rollout = torch.zeros_like(r_rollout)
    gae_tm1 = 0
    for t in reversed(range(0, len(r_rollout))):
        if t == len(r_rollout) -1:
            last_val = final_r
        else:
            last_val = ppo_model(s_rollout[t+1])[0].squeeze(-1)
        td = last_val *  0.99 + r_rollout[t] - ppo_model(s_rollout[t])[0].squeeze(-1)
        adv_rollout[t] = gae_tm1 = td * 0.99 * 0.95 * gae_tm1

    gae_rollout = adv_rollout + ppo_model(s_rollout)[0].squeeze(-1)
    #print(gae_rollout.size())
    return gae_rollout

avg_r = deque(maxlen=50)
count = 0
clip_range = 0.2

for i in range(50000):

    r_trajectory = 0

    while buffer.qty <= 200:
        a_pi = ppo_model(torch.from_numpy(s_t).float())[1]

        #a_t = int(torch.multinomial(a_pi, 1).detach())
        a_t, _, _ = select_action(a_pi)
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
        r_trajectory, _ = ppo_model(torch.from_numpy(s_tp1).float())  
    
    #begin training
    s_rollout, a_rollout, r_rollout, pi_rollout, len_rollout = buffer.pop_all()

    #Q_rollout = calc_gae(r_rollout, s_rollout, r_trajectory).unsqueeze(-1).detach()
    Q_rollout = calc_returns(r_rollout, r_trajectory).unsqueeze(-1).detach()

    V_rollout = ppo_model(s_rollout.float())[0].unsqueeze(-1)

    old_probs = Categorical(pi_rollout).log_prob(a_rollout).detach()

    for i in range(len_rollout):  
        #new outputs
        new_v, new_pi = ppo_model(s_rollout[i])
        new_dist = Categorical(new_pi)

        #critic
        #print(Q_rollout.size())
        #print(Q_rollout[i].size(), new_v.size())
        critic_loss = F.mse_loss(Q_rollout[i], new_v)
        
        #actor
        log_probs = new_dist.log_prob(a_rollout[i])
        entropy = new_dist.entropy().mean()   #ppo entropy loss
        
        adv = Q_rollout[i] - V_rollout[i].detach()

        #if len(adv) > 1: adv = (adv - adv.mean())/(adv.std() + 1e-8) #ppo minibatch advantage normalisations

        ratio = torch.exp(log_probs - old_probs[i])   #ppo policy loss function
        clipped_ratio = torch.clamp(ratio, 1-clip_range, 1+clip_range)
        actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

        loss = actor_loss + (critic_loss * 0.5) - (entropy * 0.01)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), 0.5) #ppo gradient clipping
        optimizer.step()

    

