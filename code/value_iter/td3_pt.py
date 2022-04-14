import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from gym.wrappers import RecordEpisodeStatistics
from torch.nn.utils import clip_grad_norm_
from collections import deque
from typing import NamedTuple
import random

#from code.utils.models import td3_Actor, td3_Critic
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
        lr=3e-4,
        buffer_size=1000000, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        train_after=0,
        policy_delay=2,
        action_noise=0.1,
        target_policy_noise=0.2, 
        target_noise_clip=0.5,
        verbose=500):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay
        self.action_noise = action_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.verbose=verbose
                
        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        #Critic update
        with torch.no_grad():
            a_p = self.actor_target(s_p)
            a_p = self.noisy_action(a_p, self.target_policy_noise)
            target_q1, target_q2 = self.critic_target(s_p, a_p)
            target_q = torch.min(target_q1, target_q2)
            y = r + self.gamma * target_q * (1 - d)

        q1, q2 = self.critic(s, a)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            policy_loss = -self.critic.q1_forward(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss    

    def noisy_action(self, a_t, std_amnt):
        mean=torch.zeros_like(a_t)
        noise = torch.normal(mean=mean, std=std_amnt).clamp(-self.target_noise_clip, self.target_noise_clip)
        return (a_t + noise).clamp(self.act_low,self.act_high)

    def select_action(self, s):
        a = self.actor(torch.from_numpy(s).float()).detach()
        if self.actor.training:
            a = self.noisy_action(a, self.action_noise)
        else:
            a = self.noisy_action(a, 0)
        return a.numpy()

def main():
    #env_name = 'MountainCarContinuous-v0'
    env_name = 'LunarLanderContinuous-v2'
    #env_name = 'Pendulum-v1'
    #env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    obs_dim = env.observation_space.shape[0]            
    act_dim = env.action_space.shape[0]

    episodic_rewards = deque(maxlen=10)
    episodes = 0
    r_sum = 0

    td3_agent = TD3(environment=env,
        actor=td3_Actor(obs_dim, act_dim, env.action_space.high[0]),
        critic=td3_Critic(obs_dim, act_dim), 
        lr=1e-3,
        train_after=10000,
        buffer_size=200000, 
        batch_size=100,
        gamma=0.98,
        verbose=2000)

    replay_buffer = deque(maxlen=td3_agent.buffer_size)

    s_t = env.reset()

    for i in range(300000):
        a_t = td3_agent.select_action(s_t)
                
        s_tp1, r_t, done, _ = env.step(a_t)
        r_sum += r_t
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
        s_t = s_tp1

        if len(replay_buffer) >= td3_agent.batch_size and i >= td3_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=td3_agent.batch_size)))
            loss = td3_agent.update(batch, i)

            if i % td3_agent.verbose == 0:
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_t = env.reset()
        

    """
    Render trained agent
    """
    td3_agent.actor.eval()
    s_t = env.reset()
    while True:
        env.render()
        a_t = td3_agent.select_action(s_t).numpy()
        s_tp1, r_t, done, _ = env.step(a_t)
        s_t = s_tp1
        if done:
            s_t = env.reset()        

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()