import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import random
import math
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

class DQN:
    def __init__(
        self,
        environment,
        network,
        gamma=0.99,
        train_after=50000,
        target_update=1000,
        batch_size=64,
        verbose=500,
        learning_rate=1e-4
    ):
        self.environment = environment
        self.gamma = gamma
        self.train_after = train_after
        self.target_update = target_update
        self.batch_size = batch_size
        self.q_func = network
        self.q_target = copy.deepcopy(self.q_func)
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

        self.verbose = verbose

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        #a = torch.IntTensor(batch.a)#.unsqueeze(1)  # remove for LunarLander-v2
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(1)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)

        q = torch.gather(self.q_func(s), 1, a.long())

        q_p = self.q_target(s_p).detach()

        y = r + 0.99 * q_p.max(1)[0].view(self.batch_size, 1) * (1 - d)

        loss = F.mse_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
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
    env_name = "CartPole-v0"
    #env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n    #shape[0]

    replay_buffer = deque(maxlen=1000000)

    dqn_agent = DQN(env, Q_duelling(obs_dim, act_dim))  
    #1e-3 for quicker but unstable learning for cartpole

    episodes = 0
    s_t = env.reset()

    losses = deque(maxlen=dqn_agent.verbose)
    episodic_rewards = deque(maxlen=10)

    for i in range(100000):
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        # if i % 10 == 0: print(f"Epsilon value: {dqn_agent.EPS}")

        if len(replay_buffer) >= dqn_agent.batch_size and i >= dqn_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=dqn_agent.batch_size)))
            
            if i % 4 == 0:
                loss = dqn_agent.update(batch)
                losses.append(loss)
            
            if i % dqn_agent.verbose == 0:
                avg_losses = sum(losses) / dqn_agent.verbose
                avg_r = sum(episodic_rewards) / 10
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Loss: {avg_losses} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

            if i % dqn_agent.target_update == 0:
                dqn_agent.update_target()

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

