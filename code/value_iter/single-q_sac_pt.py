#currently reaches 200+ score on LunarLander in 322 episodes/130,000 timesteps
#bit unstable but slowly gets there
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import numpy as np
import gym
import copy
from gym.wrappers import RecordEpisodeStatistics
from utils.models import ddpg_Critic, sac_Actor

from collections import deque
from typing import NamedTuple
import random

class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

class one_q_SAC():
    def __init__(self, 
        environment, 
        actor,
        critic,
        pi_lr=1e-4, 
        c_lr=1e-3, 
        buffer_size=1000000, 
        batch_size=100, 
        alpha=0.2,
        tau=0.005, 
        gamma=0.99, 
        train_after=0,
        policy_delay=2):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay

        obs_dim = environment.observation_space.shape[0]            
        act_dim = environment.action_space.shape[0]    
        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor(obs_dim, act_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.critic = critic(obs_dim, act_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        # action rescaling
        #print(self.act_high, self.act_low)
        self.action_scale = 1   #torch.IntTensor((1 - (-1)) / 2.0).float()
        self.action_bias = 0    #torch.FloatTensor((1 + (-1)) / 2.0)

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        #Critic update
        with torch.no_grad():
            a_p, log_pi, _ = self.select_action(s_p)
            target_q = self.critic_target(s_p, a_p) - log_pi * self.alpha
            y = r + self.gamma * target_q * (1 - d)

        q = self.critic(s, a)

        critic_loss = F.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            a_p, log_pi, _ = self.select_action(s)
            policy_loss = ((self.alpha * log_pi) -self.critic.forward(s, a_p)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss

    def select_action(self, s):
        mean, log_std = self.actor(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def act(self, s):
        a = self.select_action(torch.from_numpy(s).float())[0].detach().numpy()
        return a

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'  
    """
    -DDPG with regular critic learns continuous cartpole in ~20,000 timesteps with train_after=10,000
    -adding encoder to the network doesnt completely hinder learning but slows it down a 
        bit and makes it unstable, maybe tweak the size of the encoder
    -a lot to talk about in terms of experimenting with how you transferred ciq to ddpg
    """
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    c_losses = deque(maxlen=100)
    episodic_rewards = deque(maxlen=10)
    episodes = 0

    sac_agent = one_q_SAC(environment=env,    #taken from sb3 zoo
        actor=sac_Actor,
        critic=ddpg_Critic,
        buffer_size=200000, 
        gamma=0.98, 
        train_after=10000)

    replay_buffer = deque(maxlen=sac_agent.buffer_size)

    s_t = env.reset()

    for i in range(300000):
        a_t = sac_agent.act(s_t)
        
        s_tp1, r_t, done, info = env.step(a_t)
        
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        if len(replay_buffer) >= 100 and i > sac_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=100)))
            loss = sac_agent.update(batch, i)

            if i % 1000 == 0: 
                c_losses.append(loss)
                avg_c_losses = sum(c_losses)/100    #for formatting, I want to round it better than just making it an int!
                avg_r = sum(episodic_rewards)/10
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Critic Loss: {int(avg_c_losses)} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(int(info['episode']['r']))
            s_tp1 = env.reset()

        s_t = s_tp1

    #Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = sac_agent.act(s_t)
        s_tp1, r_t, done, _ = env.step(a_t)
        if done:
            s_tp1 = env.reset()

        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()