import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from gym.wrappers import RecordEpisodeStatistics

from utils.models import Actor, td3_Critic
#from utils.memory import Memory

"""
implementations for help:
https://github.com/sfujim/TD3/blob/master/TD3.py
https://colab.research.google.com/drive/19-z6Go3RydP9_uhICh0iHFgOXyKXgX7f#scrollTo=zuw702kdRr4E
"""

from collections import deque
from typing import NamedTuple
import random
class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

def noisy_action(a_t, env):
    noise = (torch.randn_like(a_t) * 0.2).clamp(-0.5, 0.5)
    return (a_t + noise).clamp(env.action_space.low[0],env.action_space.high[0])

def update(batch, i):
    s = torch.from_numpy(np.array(batch.s))
    a = torch.from_numpy(np.array(batch.a))#.unsqueeze(1)    #remove for LunarLanderContinuous-v2
    r = torch.FloatTensor(batch.r).unsqueeze(1)
    s_p = torch.from_numpy(np.array(batch.s_p))
    d = torch.IntTensor(batch.d).unsqueeze(1)

    #print(f"s dims: {s.size()}, a dims: {a.size()}")

    #Critic
    with torch.no_grad():
        a_p = actor(s_p)
        target_q1, target_q2 = critic_target(s_p, a_p)
        target_q = torch.min(target_q1, target_q2)
        y = r + 0.99 * target_q * (1 - d)

    q1, q2 = critic(s, a)

    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    #Actor
    if i % 2 == 0:
        policy_loss = -critic.q1_forward(s, actor(s)).mean()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

    return critic_loss

#env_name = 'MountainCarContinuous-v0'
#env_name = 'LunarLanderContinuous-v2'
env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = RecordEpisodeStatistics(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

replay_buffer = deque(maxlen=100000)

actor = Actor(obs_dim, act_dim, env.action_space.high[0])
actor_target = copy.deepcopy(actor)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

critic = td3_Critic(obs_dim, act_dim)
critic_target = copy.deepcopy(critic)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

s_t = env.reset()
tau = 0.005

c_losses = []
episodic_rewards = []

for i in range(300000):
    a_t = actor(torch.from_numpy(s_t)).detach()
    a_t = np.array(noisy_action(a_t, env))
    
    s_tp1, r_t, done, info = env.step(a_t)
    
    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if len(replay_buffer) >= 128:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        loss = update(batch, i)
        c_losses.append(loss)
        avg_c_losses = (sum(c_losses[-100:]))/100
        avg_r = (sum(episodic_rewards[-100:]))/100

        if i % 100 == 0: print(f"Timestep: {i} | Avg. Critic Loss: {avg_c_losses} | avg. Reward: {avg_r}")        

        if i % 2 == 0:
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
            
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    if done:
        episodic_rewards.append(int(info['episode']['r']))
        s_tp1 = env.reset()

    s_t = s_tp1

#Render Trained agent
s_t = env.reset()
while True:
    env.render()
    a_t = actor(torch.from_numpy(s_t)).detach()
    s_tp1, r_t, done, _ = env.step(a_t)
    if done:
        s_tp1 = env.reset()

    s_t = s_tp1