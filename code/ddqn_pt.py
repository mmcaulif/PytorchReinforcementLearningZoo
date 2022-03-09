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

from utils.models import Q_val, Q_duelling
class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int

class DDQN:
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

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

        self.verbose = verbose

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)

        q = self.q_func(s).gather(1, a.long())

        with torch.no_grad():
            a_p = torch.argmax(self.q_func(s_p), dim = 1).unsqueeze(1)  #actions selected from local q
            q_p = self.q_target(s_p).gather(1, a_p)    #actions evaluated from target q
            y = r + self.gamma * q_p * (1 - d)
            
        loss = F.mse_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def update_target(self):
        self.q_target = copy.deepcopy(self.q_func)

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            a = torch.argmax(self.q_func(torch.from_numpy(s)).detach()).numpy()
        else:
            a = self.environment.action_space.sample()

        return a

def main():
    #env_name = "CartPole-v0"
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n    #shape[0]

    replay_buffer = deque(maxlen=10000)

    #net = Q_val(obs_dim, act_dim)
    net = Q_duelling(obs_dim, act_dim)
    dqn_agent = DDQN(env, net, train_after=250, target_update=300, batch_size=32, verbose=2000, learning_rate=0.004)    #hyperparameters for lunarlander

    episodes = 0
    s_t = env.reset()

    episodic_rewards = deque(maxlen=20)

    for i in range(1000000):
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        if len(replay_buffer) >= dqn_agent.batch_size and i >= dqn_agent.train_after:
            
            if i % dqn_agent.train_freq == 0:
                batch = Transition(*zip(*random.sample(replay_buffer, k=dqn_agent.batch_size)))
                loss = dqn_agent.update(batch)

            if i % dqn_agent.target_update == 0:
                dqn_agent.update_target()
            
            if i % dqn_agent.verbose == 0:
                avg_r = sum(episodic_rewards) / len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(int(info["episode"]["r"]))
            s_tp1 = env.reset()

        s_t = s_tp1

    # Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        if done:
            print(f'Episode Complete, reward = {info["episode"]["r"]}')
            s_tp1 = env.reset()

        s_t = s_tp1    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

