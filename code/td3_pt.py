from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from gym.wrappers import RecordEpisodeStatistics

from utils.models import td3_Actor, td3_Critic
#from pytorch_continous_RL.code.utils.models import td3_Actor, td3_Critic

#from utils.memory import Memory

"""
implementations for help:
https://github.com/sfujim/TD3/blob/master/TD3.py
https://colab.research.google.com/drive/19-z6Go3RydP9_uhICh0iHFgOXyKXgX7f#scrollTo=zuw702kdRr4E

To do:
-add LR schedule
-gradient clipping
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

class TD3():
    def __init__(self, 
        environment, 
        actor,
        critic,
        pi_lr=1e-4, 
        c_lr=1e-3, 
        buffer_size=1000000, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        policy_delay=2,
        target_policy_noise=0.2, 
        target_noise_clip=0.5,
        debug_dim=[],
        debug_act_high=[],
        debug_act_low=[]):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        try:
            obs_dim = environment.observation_space.shape[0]            
            act_dim = environment.action_space.shape[0]    
            self.act_high = environment.action_space.high[0]
            self.act_low = environment.action_space.low[0]    
        except:
            obs_dim = debug_dim[0]
            act_dim = debug_dim[1]
            self.act_high = torch.tensor(debug_act_high)
            self.act_low = torch.tensor(debug_act_low)

        self.actor = actor(obs_dim, act_dim, self.act_high)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.critic = critic(obs_dim, act_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def noisy_action(self, a_t):
        noise = (torch.randn_like(a_t) * self.target_noise_clip).clamp(-self.target_policy_noise, self.target_policy_noise)
        return (a_t + noise).clamp(self.act_low,self.act_high)

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)

        #Critic update
        with torch.no_grad():
            a_p = self.actor(s_p)
            target_q1, target_q2 = self.critic_target(s_p, a_p)
            target_q = torch.min(target_q1, target_q2)
            y = r + self.gamma * target_q * (1 - d)

        q1, q2 = self.critic(s, a)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            policy_loss = -self.critic.q1_forward(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss

def main():
    #env_name = 'MountainCarContinuous-v0'
    env_name = 'LunarLanderContinuous-v2'
    #env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    replay_buffer = deque(maxlen=1000000)

    c_losses = deque(maxlen=100)
    episodic_rewards = deque(maxlen=10)
    episodes = 0

    td3_agent = TD3(environment=env, actor=td3_Actor, critic=td3_Critic)

    s_t = env.reset()

    for i in range(100000):
        a_t = td3_agent.actor(torch.from_numpy(s_t)).detach()
        a_t = np.array(td3_agent.noisy_action(a_t))
        
        s_tp1, r_t, done, info = env.step(a_t)
        
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        if len(replay_buffer) >= 100:
            batch = Transition(*zip(*random.sample(replay_buffer, k=100)))
            loss = td3_agent.update(batch, i)

            if i % 100 == 0: 
                c_losses.append(loss)
                avg_c_losses = sum(c_losses)/100
                avg_r = sum(episodic_rewards)/10
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Critic Loss: {avg_c_losses} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(int(info['episode']['r']))
            s_tp1 = env.reset()

        s_t = s_tp1

    #Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = np.array(td3_agent.actor(torch.from_numpy(s_t)).detach())
        s_tp1, r_t, done, _ = env.step(a_t)
        if done:
            s_tp1 = env.reset()

        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()