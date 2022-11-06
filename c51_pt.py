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
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

from code.utils.models import Q_dist

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int

class C51:
    def __init__(
        self,
        env,
        network,
        gamma=0.99,
        train_after=50000,
        train_freq=4,
        target_update=1000,
        batch_size=64,
        learning_rate=1e-4,
        max_grad_norm=10,
        n_atoms=51,
        v_max=10,
        v_min=-10
    ):
        self.env = env
        self.q_func = network(env.observation_space.shape[0], env.action_space.n, n_atoms)
        self.q_target = copy.deepcopy(self.q_func)
        self.gamma = gamma
        self.train_after = train_after
        self.train_freq = train_freq
        self.target_update = target_update
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.n_atoms = n_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.dt_z = (self.v_max - self.v_min)/(self.n_atoms - 1)

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1).type(torch.float32)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)
            
        loss = 0 # to be replaced by KL divergence

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            a_dist = torch.zeros(self.n_atoms)
            # replaced with whatever way c51 chooses actions
        else:
            a = self.env.action_space.sample()

        return a, a_dist

    def train(self, train_steps, r_avg_len=20, verbose=500):
        episodes = 0
        episodic_rewards = deque(maxlen=r_avg_len)
        r_sum = 0
        replay_buffer = deque(maxlen=100000)

        # Training loop
        s_t = self.env.reset()
        for i in range(train_steps):
            a_t = self.select_action(s_t)
            s_tp1, r_t, done, _ = self.env.step(a_t)
            r_sum += r_t
            replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
            s_t = s_tp1

            if len(replay_buffer) >= self.batch_size and i >= self.train_after:
                
                if i % self.train_freq == 0:
                    batch = Transition(*zip(*random.sample(replay_buffer, k=self.batch_size)))
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

    c51_agent = C51(
        env,
        Q_dist, 
        train_after=250,
        target_update=300,
        learning_rate=0.001)    #hyperparameters for lunarlander

    s_t = env.reset()
    out = c51_agent.q_func(s_t)
    print(out.shape)
    #print(out[0])
    
    #c51_agent.train(train_steps=50000, r_avg_len=5, verbose=500)
    
    sys.exit()           

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

