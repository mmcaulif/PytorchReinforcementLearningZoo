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
    for t in reversed(range(len(r_rollout))):
        final_r = final_r *  0.99 + r_rollout[t]
        discounted_r[t] = final_r
    return discounted_r

def calc_gae(r_rollout, s_rollout, d_rollout, final_r): #Still need to understand!
    gamma = 0.98
    lmbda = 0.8
    gae_returns = torch.zeros_like(r_rollout)
    gae = 0
    for t in reversed(range(len(r_rollout))):
        if t == len(r_rollout) - 1: v_tp1 = final_r
        else: v_tp1 = ppo_model(s_rollout[t+1])[0].squeeze(-1)

        v_t = ppo_model(s_rollout[t])[0].squeeze(-1)
        y = r_rollout[t] + v_tp1 * gamma * (1 - d_rollout[t])
        td = y - v_t
        gae = td + gamma * lmbda * gae * (1 - d_rollout[t])

        gae_returns[t] = gae + v_t

    return gae_returns

avg_r = deque(maxlen=50)
count = 0
clip_range = 0.2
batch_size = 1

for i in range(50000):

    r_trajectory = 0

    while buffer.qty <= 2048:
        a_pi = ppo_model(torch.from_numpy(s_t).float())[1]

        #a_t = int(torch.multinomial(a_pi, 1).detach())
        a_t, _, _ = select_action(a_pi)
        s_tp1, r, d, info = env.step(a_t)
        buffer.push(s_t, a_t, r, a_pi, d)
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
    s_rollout, a_rollout, r_rollout, pi_rollout, d_rollout, len_rollout = buffer.pop_all()

    #Q_rollout = calc_returns(r_rollout, r_trajectory).detach()
    Q_rollout = calc_gae(r_rollout, s_rollout, d_rollout, r_trajectory).detach()
    #print('Returns: ', Q_rollout, '\nGAE Returns: ', gae_rollout, '\n')

    V_rollout = ppo_model(s_rollout.float())[0]

    old_probs = Categorical(pi_rollout).log_prob(a_rollout).detach()
    indexes = np.arange(len_rollout)

    for iter in range(0, len_rollout, batch_size):  
        iter_end = iter + batch_size
        i = indexes[iter:iter_end]

        #new outputs
        new_v, new_pi = ppo_model(s_rollout[iter:iter_end])
        new_dist = Categorical(new_pi)

        #critic
        critic_loss = F.mse_loss(Q_rollout[iter:iter_end], new_v.squeeze(-1))
        
        #actor
        log_probs = new_dist.log_prob(a_rollout[iter:iter_end])
        entropy = new_dist.entropy().mean()   #ppo entropy loss
        
        adv = Q_rollout[iter:iter_end] - V_rollout[iter:iter_end].detach()

        #if len(adv) > 1: adv = (adv - adv.mean())/(adv.std() + 1e-8) #ppo minibatch advantage normalisations

        ratio = torch.exp(log_probs - old_probs[iter:iter_end])   #ppo policy loss function
        clipped_ratio = torch.clamp(ratio, 1-clip_range, 1+clip_range)
        actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

        loss = actor_loss + (critic_loss * 0.5) - (entropy * 0.00)  #zoo entropy coeff for cartpole

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), 0.5) #ppo gradient clipping
        optimizer.step()

    

