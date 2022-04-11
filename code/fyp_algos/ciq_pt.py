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
import wandb

#https://stackoverflow.com/questions/24622041/python-importing-a-module-from-a-parallel-directory
#example:   python -m code.fyp_algos.ciq_pt

from code.value_iter.ddqn_pt import DDQN
from code.utils.models import Q_val
from code.utils.attacker import Attacker

class Q_ciq(nn.Module):
    def __init__(self, num_treatment=4, act_dims=2, obs_dims=4):
        super(Q_ciq, self).__init__()
        
        self.encoder = nn.Sequential(nn.Linear(obs_dims, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     #nn.ReLU(),    #layer added for debugging
                                     )

        self.logits_t = nn.Sequential(nn.Linear(64, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, num_treatment)
                                      )
        
        self.fc = nn.Sequential(nn.Linear((64 + num_treatment), (64 + num_treatment)),
                                nn.ReLU(),
                                nn.Linear((64 + num_treatment), act_dims)
                                )

    def forward(self, s, t_labels):
        z = self.encoder(s)  #comes out as a flattened tensor of length 128 (step * 32)
        t_p = self.logits_t(z)  #outputs as a step * num treatments tensor
        
        if self.training:
            q = self.fc(torch.cat([z, t_labels], dim=-1))
        else:
            q = self.fc(torch.cat([z, t_p], dim=-1))

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
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1).type(torch.float32)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_t = torch.from_numpy(np.array(batch.t)).type(torch.float32)

        q, i_p = self.q_func(s, i_t)
        q = q.gather(1, a.long())

        with torch.no_grad():
            i_ghost = torch.zeros(i_t.size())
            a_p = torch.argmax(self.q_func(s_p, i_ghost)[0], 1).unsqueeze(1)
            q_p = self.q_target(s_p, i_ghost)[0].gather(1, a_p)
            y = r + self.gamma * q_p * (1 - d)

        i_loss = F.binary_cross_entropy_with_logits(i_p, i_t)
        loss = F.mse_loss(q, y) + i_loss     

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss, i_loss

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def soft_update(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            q = self.q_func(torch.from_numpy(s).type(torch.float32), torch.Tensor([1,0,0,0]))[0].detach()
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
    #wandb.init(project="fyp-ciq", entity="manusft")
    P = 0.3
    vanilla = False
    env_name = 'CartPole-v0'

    """wandb.config = {
        "env_name": env_name,
        "P": P,
        "Vanilla q network": vanilla
    }"""

    env = gym.make(env_name)
    env = Attacker(env, p=P)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n    #shape[0]
    
    if vanilla:
        print("Using vanilla ddqn")
        ciq_agent = DDQN(env, 
            Q_val(obs_dim, act_dim), 
            train_after=1000, 
            target_update=10, 
            batch_size=64, 
            learning_rate=5e-4) #fails for 1e-4, suceeds with >3e-4
    else:
        print("Using ciq-ddqn") ###!!! (bs=128, lr=5e-4) hyperparams get to 200 ~20,000 steps with P=0.05
        ciq_agent = CIQ(env, 
            Q_ciq(), 
            train_after=1000, 
            target_update=10, 
            batch_size=256, 
            learning_rate=5e-4) #zoo uses 2.3e-3, suceeds for 1e-3, 5e-4

    replay_buffer = deque(maxlen=100000)

    episodic_rewards = deque(maxlen=10)
    r_sum = 0
    episodes = 0
    s_t = env.reset()

    for i in range(40000+1):
        #a_t = env.action_space.sample()
        a_t = ciq_agent.select_action(s_t)
        s_tp1, r_t, done, i_t = env.step(a_t)
        r_sum += r_t
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done, i_t])

        if len(replay_buffer) >= ciq_agent.batch_size and i >= ciq_agent.train_after:
            
            if i % ciq_agent.train_freq == 0:
                batch = Transition(*zip(*random.sample(replay_buffer, k=ciq_agent.batch_size)))
                loss = ciq_agent.update(batch)
                #ciq_agent.soft_update()
                
            if i % ciq_agent.target_update == 0:
                ciq_agent.hard_update()
                
            if i % ciq_agent.verbose == 0 and i > 0:
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")            

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_tp1 = env.reset()

        s_t = s_tp1
    
        #if i % 1000 == 0 and i > 0:
            #wandb.log({f"Average episodic reward, P={P}":torch.Tensor(episodic_rewards).mean()})
            #wandb.log({"dqn long train":torch.Tensor(episodic_rewards).mean()})

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

