from turtle import done
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
            nn.Linear(input_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )
        self.actor = torch.nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

    def value(self, s):
        out = self.critic(s)
        return out

    def policy(self, s):
        out = self.actor(s)
        return out

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.rewards, self.policies = [], [], [], []
    
    def push(self, state, action, reward, policy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
    
    def pop_all(self):
        states = torch.as_tensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1)
        policies = torch.stack(self.policies).float()
        
        self.states, self.actions, self.rewards, self.policies = [], [], [], []
        
        return states, actions, rewards.squeeze(-1), policies

    
def select_action(x):
    x_t = int(torch.multinomial(x, 1).detach())
    x_dist = Categorical(x)
    return x_t, x_dist.log_prob(torch.tensor(x_t)), x_dist.entropy().mean()

def rollout(s_t, n_steps, buffer):
    final_r = 0

    for i in range(n_steps):
        a_pi = policy(torch.from_numpy(s_t).float())
        a_t = int(torch.multinomial(a_pi, 1).detach())
        s_tp1, r, d, info = env.step(a_t)
        buffer.push(s_t, a_t, r, a_pi)
        if d:
            s_tp1 = env.reset()
            print(f'Episode reward: {int(info["episode"]["r"])}')
            break

        s_t = s_tp1

    if not d:
        final_r = value(torch.from_numpy(s_tp1).float())  

    return s_t, buffer, final_r

env = RecordEpisodeStatistics(gym.make("CartPole-v1"))
s_t = env.reset()

a2c_model = Model(4, 2)
policy = a2c_model.policy
value = a2c_model.value
optimizer = torch.optim.RMSprop(a2c_model.parameters(), lr=7e-3, eps=1e-5)  #should be lr=7e-3
buffer = Memory()

for i in range(100000):
    #env.render()
    
    s_t, buffer, R = rollout(s_t, n_steps=5, buffer=buffer)
    
    s_rollout, a_rollout, r_rollout, pi_rollout = buffer.pop_all()
    
    r_trajectory = torch.zeros_like(r_rollout)

    for t in reversed(range(0, len(r_rollout))):
        R *= 0.99
        R += r_rollout[t]
        r_trajectory[t] = R
        
    V = value(s_rollout.float()).squeeze(-1)
    
    """print('Calculated return: ', r_trajectory.detach())
    print('Critic estimates: ',V.detach(), '\n')"""
    
    critic_loss = F.mse_loss(r_trajectory, V)
    #print(critic_loss)
    
    dist_rollout = Categorical(pi_rollout)   

    entropy = dist_rollout.entropy().mean()
    adv = R - V
    log_probs = dist_rollout.log_prob(a_rollout)
    actor_loss = -(log_probs * adv.detach()).mean()

    loss = actor_loss + (critic_loss * 0.5) - (entropy * 0.01)
    #loss = critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(a2c_model.parameters(), 0.5)
    optimizer.step()

    

