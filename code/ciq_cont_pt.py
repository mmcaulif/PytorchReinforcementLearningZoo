from collections import deque
import copy
import random
from turtle import done
from typing import NamedTuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from gym import Wrapper
import wandb

from ciq_pt import GaussianNoise, Transition

from utils.models import td3_Actor, td3_Critic
from td3_pt import TD3

class ciq_Critic1(nn.Module):
    def __init__(self, step, num_treatment, obs_dims, act_dims):
        super(ciq_Critic1, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(obs_dims+act_dims, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     #nn.ReLU(),
                                     #nn.Linear(64, 64)
                                     )

        self.logits_t = nn.Sequential(nn.Linear(64, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, num_treatment)
                                      )
        
        self.fc1 = nn.Sequential(nn.Linear((64 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step, act_dims)
                                )
        
        self.fc2 = nn.Sequential(nn.Linear((64 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step, act_dims)
                                )

    def forward(self, s, a, t_labels):
        sa = torch.cat([s, a], 1)
        z = self.encoder(sa)
        t_p = self.logits_t(z)
        
        if self.training:
            q1 = self.fc1(torch.cat([z, t_labels], dim=-1))
            q2 = self.fc2(torch.cat([z, t_labels], dim=-1))
        else:
            q1 = self.fc1(torch.cat([z, t_labels], dim=-1))
            q2 = self.fc2(torch.cat([z, t_labels], dim=-1))

        return [q1,q2], t_p

    def q1_forward(self, s, a, t_labels):
        sa = torch.cat([s, a], 1)
        z = self.encoder(sa)
        t_p = self.logits_t(z)
        
        if self.training:
            q1 = self.fc1(torch.cat([z, t_labels], dim=-1))
        else:
            q1 = self.fc1(torch.cat([z, t_labels], dim=-1))

        return q1, t_p

class ciq_Critic2(nn.Module):
    def __init__(self, step, num_treatment, obs_dims, act_dims):
        super(ciq_Critic2, self).__init__()

        self.logits_t = nn.Sequential(nn.Linear(obs_dims+act_dims, 64),
                                      nn.ReLU(),
                                      nn.Linear(64,64),
                                      nn.ReLU(),
                                      nn.Linear(64, num_treatment)
                                      )
        
        self.fc1 = nn.Sequential(nn.Linear((obs_dims + act_dims + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step, 1)
                                )
        
        self.fc2 = nn.Sequential(nn.Linear((obs_dims + act_dims + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step , (256 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((256 + num_treatment) * step, 1)
                                )

    def forward(self, s, a, t_labels):
        sa = torch.cat([s, a], 1)
        t_p = self.logits_t(sa)
        
        if self.training:
            q1 = self.fc1(torch.cat([sa, t_labels], dim=-1))
            q2 = self.fc2(torch.cat([sa, t_labels], dim=-1))
        else:
            q1 = self.fc1(torch.cat([sa, t_labels], dim=-1))
            q2 = self.fc2(torch.cat([sa, t_labels], dim=-1))

        return [q1, q2], t_p

    def q1_forward(self, s, a, t_labels):
        sa = torch.cat([s, a], 1)
        t_p = self.logits_t(sa)
        
        if self.training:
            q1 = self.fc1(torch.cat([sa, t_labels], dim=-1))
        else:
            q1 = self.fc1(torch.cat([sa, t_labels], dim=-1))

        return q1, t_p

class CIQ_cont():
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
        train_after=0,
        policy_delay=2,
        exploration_noise=0.1,
        policy_noise=0.2, 
        target_noise_clip=0.5,
        verbose=500,
        EPS_END=0.5
        ):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.target_noise_clip = target_noise_clip

        """obs_dim = environment.observation_space.shape[0]            
        act_dim = environment.action_space.shape[0]"""
        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.verbose = verbose

        self.EPS_END = EPS_END
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_t = torch.from_numpy(np.array(batch.t)).type(torch.float32)

        #Critic update
        with torch.no_grad():
            i_ghost = torch.zeros(i_t.size())
            a_p = self.actor_target(s_p)
            a_p = torch.from_numpy(self.noisy_action(a_p, self.policy_noise))
            target_q1, target_q2 = self.critic_target(s_p, a_p, i_ghost)[0]
            target_q = torch.min(target_q1, target_q2)
            y = r + self.gamma * target_q * (1 - d)

        (q1, q2), i_p = self.critic(s, a, i_t)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y) + F.binary_cross_entropy_with_logits(i_p, i_t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            policy_loss = -self.critic.q1_forward(s, self.actor(s), i_t)[0].mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss 
    
    def noisy_action(self, a_t, amount):
        mean=torch.zeros_like(a_t)
        noise = torch.normal(mean=mean, std=amount).clamp(-self.target_noise_clip, self.target_noise_clip)
        return (a_t + noise).clamp(self.act_low,self.act_high).numpy()

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            #a = self.actor(torch.tensor(s).float()).detach()
            a = self.actor(torch.from_numpy(s).type(torch.float32)).detach()    #might be faster?
        else:
            a = torch.from_numpy(self.environment.action_space.sample()).type(torch.float32)

        return a

def main():
    
    P = 0.0
    vanilla = True
    env_name = 'LunarLanderContinuous-v2'
    """wandb.init(project="fyp-ciq", entity="manusft")
    wandb.config = {
        "env_name": env_name,
        "P": P,
        "Vanilla q network": vanilla
    }"""
    
    env = gym.make(env_name)
    env = GaussianNoise(env, p=P, var=0.0) #at p >= 0.1, learning is already stunted for vanilla dqn
    
    if vanilla:
        ciq_agent = TD3(environment=env,    #taken from sb3 zoo
            actor=td3_Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0]),
            critic=td3_Critic(env.observation_space.shape[0], env.action_space.shape[0]),
            gamma=0.98,
            train_after=10000,
            target_policy_noise=0.1,
            verbose=2000,
            EPS_END=0.2)

    else:
        ciq_agent = CIQ_cont(environment=env,    #taken from sb3 zoo
            actor=td3_Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0]),
            critic=ciq_Critic2(1, 2, env.observation_space.shape[0], env.action_space.shape[0]), 
            buffer_size=200000, 
            gamma=0.98, 
            train_after=10000,
            target_policy_noise=0.1,
            verbose=2000,
            EPS_END=0.2)

    replay_buffer = deque(maxlen=ciq_agent.buffer_size)

    episodic_rewards = deque(maxlen=30)
    r_sum = 0
    episodes = 0
    s_t = env.reset()

    for i in range(300000):        
        a_t = ciq_agent.actor(torch.from_numpy(s_t).float()).detach()
        a_t = ciq_agent.noisy_action(a_t)

        s_tp1, r_t, done, i_t = env.step(a_t)
        r_sum += r_t

        replay_buffer.append([s_t, a_t, r_t, s_tp1, done, i_t])

        if len(replay_buffer) >= ciq_agent.batch_size and i >= ciq_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=ciq_agent.batch_size)))
            loss = ciq_agent.update(batch, i)
                
            if i % ciq_agent.verbose == 0 and i > 0:
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]") 
                #wandb.log({"td3_ciq train":avg_r})           

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_tp1 = env.reset()
        
        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()