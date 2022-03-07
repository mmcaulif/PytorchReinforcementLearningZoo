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

class GaussianNoise(Wrapper):
    def __init__(self, env, p=0.1, var=0.5):
        super().__init__(env)
        self.env = env
        self.p = p
        self.var = var
        self.obs_low = env.observation_space.low[0]
        self.obs_high = env.observation_space.high[0]

    def step(self, action):  
        next_state, reward, done_bool, _ = super().step(action)
        t_labels = np.zeros(2)
        noise = torch.normal(mean=torch.zeros_like(torch.from_numpy(next_state)), std=self.var).numpy()
        
        I = 0
        if torch.rand(1) < self.p: I = 1
        t_labels[I] = 1
        gaussian_state = np.clip((next_state + noise), self.obs_low, self.obs_high)
        next_state = I * gaussian_state + (1 - I) * next_state

        return next_state, reward, done_bool, t_labels

class Q_ciq(nn.Module):
    def __init__(self, step=4, num_treatment=2, act_dims=2, obs_dims=4):
        super(Q_ciq, self).__init__()
        self.step = step, 
        self.num_treatment = num_treatment,
        
        self.encoder = nn.Sequential(nn.Linear(obs_dims, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),    #layer added for debugging
                                     )

        self.logits_t = nn.Sequential(nn.Linear(64, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, num_treatment)
                                      )
        
        self.fc = nn.Sequential(nn.Linear((64 + num_treatment) * step , (64 + num_treatment) * step),
                                nn.ReLU(),
                                nn.Linear((64 + num_treatment) * step, act_dims)
                                )

    def forward(self, s, t_labels):
        z = self.encoder(s)  #comes out as a flattened tensor of length 128 (step * 32)
        t_p = self.logits_t(z)  #outputs as a step * num treatments tensor
        q = self.fc(torch.cat([z, t_labels], dim=-1))
        return q, t_p

class CIQ():
    def __init__(
        self,
        environment,
        network,
        gamma=0.99,
        train_after=50000,
        train_freq=4,
        target_update=1000,
        batch_size=64,
        verbose=500,
        learning_rate=1e-4,
        max_grad_norm=10,
        tau=5e-3
    ):
        self.environment = environment
        self.q_func = network
        self.q_target = copy.deepcopy(self.q_func)
        self.gamma = gamma
        self.train_after = train_after
        self.train_freq = train_freq
        self.target_update = target_update
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.tau = tau

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

        self.verbose = verbose
        pass

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_t = torch.from_numpy(np.array(batch.t)).type(torch.float32)

        q, i_p = self.q_func(s, i_t)
        q = q.gather(1, a.long())

        with torch.no_grad():
            i_ghost = torch.zeros(i_t.size())
            a_p = torch.argmax(self.q_func(s_p, i_ghost)[0], 1).unsqueeze(1)
            q_p = self.q_target(s_p, i_ghost)[0].gather(1, a_p)
            y = r + self.gamma * q_p * (1 - d)

        loss = F.mse_loss(q, y)# +  F.binary_cross_entropy_with_logits(i_p, i_t)        

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def soft_update(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            q = self.q_func(torch.from_numpy(s), torch.Tensor([1,0]))[0].detach()
            a = torch.argmax(q).numpy()
        else:
            a = self.environment.action_space.sample()

        return a

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int
    t: int

def main():
    wandb.init(project="fyp-ciq", entity="manusft")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env = GaussianNoise(env, p=0.0) #at p = 0.1, learning is already stunted
    
    ciq_agent = CIQ(env, Q_ciq(step=1)) #ciq paper is batch_size=32 and learning_rate=5e-4
    replay_buffer = deque(maxlen=1000000)

    episodic_rewards = deque(maxlen=10)
    r_sum = 0
    episodes = 0
    s_t = env.reset()

    for i in range(200000):
        #a_t = env.action_space.sample()
        a_t = ciq_agent.select_action(s_t)
        s_tp1, r_t, done, i_t = env.step(a_t)
        r_sum += r_t
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done, i_t])

        if len(replay_buffer) >= ciq_agent.batch_size and i >= ciq_agent.train_after:
            
            if i % ciq_agent.train_freq == 0:
                batch = Transition(*zip(*random.sample(replay_buffer, k=ciq_agent.batch_size)))
                ciq_agent.update(batch)
                ciq_agent.soft_update()
                
            #if i % ciq_agent.target_update == 0:
            #    ciq_agent.hard_update()
                
            if i % ciq_agent.verbose == 0 and i > 0:
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")            

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_tp1 = env.reset()

        s_t = s_tp1
    
        if i % 1000 == 0 and i > 0:
            wandb.log({"Average episodic reward":torch.Tensor(episodic_rewards).mean()})

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

