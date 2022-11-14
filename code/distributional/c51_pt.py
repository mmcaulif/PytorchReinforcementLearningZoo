import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import gym
import copy
import random
from collections import deque
from typing import NamedTuple

from code.utils.models import Q_dist
from code.utils.memory import ReplayBuffer

class C51:
    def __init__(
        self,
        env,
        network,
        replay_buffer=ReplayBuffer,
        gamma=0.99,
        train_after=50000,
        train_freq=4,
        target_update=1000,
        batch_size=64,
        learning_rate=1e-4,
        max_grad_norm=10,
        n_atoms=51,
        v_max=200,  # for cartpole
        v_min=0
    ):
        self.env = env
        self.replay_buffer = replay_buffer(100000)
        self.q_func = network(env.observation_space.shape[0], env.action_space.n, n_atoms)
        self.q_target = copy.deepcopy(self.q_func)
        self.gamma = gamma
        self.train_after = train_after
        self.train_freq = train_freq
        self.target_update = target_update
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm

        # distributional variables
        self.n_atoms = n_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.dt_z = (self.v_max - self.v_min)/(self.n_atoms - 1)
        self.z_interval = torch.tensor([np.round(v_min + self.dt_z*i, 1) for i in range(n_atoms)]).float()

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1).type(torch.int64)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        z = self.q_func(s)
        z_p = self.q_target(s_p)
        a_p = torch.sum(torch.mul(z_p, self.z_interval), dim=-1).argmax(-1).unsqueeze(-1)
        q_p = torch.matmul(z_p, self.z_interval).gather(1, a_p)
        # ent_p = self.q_func.log_pi(s).squeeze()
        
        # print(ent_p, '/n')
        # print(a, ent_p.gather(0, a))    # no idea why it doesnt gather the right one

        ### Training steps:
        # 1. scale target distribution with gamma
        # 2. shift with r
        # 3. project new target dist over main dist
        # 4. get kl divergence / cross entropy

        m = torch.zeros(self.batch_size, self.n_atoms)
        if d == 0:
            for j in range(self.n_atoms):
                tz = (r + (self.gamma * self.z_interval[j])).clamp(self.v_min, self.v_max)
                bj = (tz - self.v_min)/self.dt_z
                l, u = bj.floor().long(), bj.ceil().long()
                print(l, u)
                m[l] += q_p[j]*(u - bj)
                m[u] += q_p[j]*(bj - l)

        print(m)
        loss = 0

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        self.EPS = 0
        if torch.rand(1) > self.EPS:
            a_dist = self.q_func(torch.from_numpy(s).float()).detach()
            act_sum = torch.matmul(a_dist, self.z_interval)
            #act_sum = torch.sum(az_dist, dim=-1)
            a = torch.argmax(act_sum).numpy()
        else:
            a = self.env.action_space.sample()

        return a

    def train(self, train_steps, r_avg_len=20, verbose=500):
        episodes = 0
        episodic_rewards = deque(maxlen=r_avg_len)
        r_sum = 0

        # Training loop
        s_t = self.env.reset()
        for i in range(train_steps):
            a_t = self.select_action(s_t)
            s_tp1, r_t, done, _ = self.env.step(a_t)
            r_sum += r_t
            self.replay_buffer.append(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

            if len(self.replay_buffer) >= self.batch_size and i >= self.train_after:
                
                if i % self.train_freq == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.update(batch)

                if i % self.target_update == 0:
                    self.hard_update()
                
                if i % verbose == 0:
                    avg_r = sum(episodic_rewards) / len(episodic_rewards)
                    print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

            if done:
                episodes += 1
                episodic_rewards.append(r_sum)
                r_sum = 0
                s_t = self.env.reset()

def main():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    c51_agent = C51(env, Q_dist, gamma=1, train_after=100, batch_size=1)

    c51_agent.train(1000)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

