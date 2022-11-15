import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import numpy as np
import gym
import copy
import random
from collections import deque
from gym.wrappers import RecordEpisodeStatistics

from code.utils.models import Q_val, Q_duelling

class Rollout_Memory(object):

    def __init__(self):
        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []
        self.qty = 0
    
    def push(self, state, action, reward, next_states, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_states)
        self.dones.append(done)
        self.qty += 1
    
    def pop_all(self):
        states = torch.as_tensor(np.array(self.states)).float()
        actions = torch.as_tensor(np.array(self.actions)).float()
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.as_tensor(np.array(self.next_states)).float()
        dones = torch.IntTensor(self.dones)
        qty = self.qty
        
        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []
        self.qty = 0
        
        return states, actions, rewards, next_states, dones, qty

class Policy(torch.nn.Module):
    def __init__(self, input_size, action_size):
        super(Policy, self).__init__()

        self.actor = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, s):
        pi = self.actor(s)
        return pi

    def get_dist(self, s):
        dist = Categorical(self.forward(s))
        return dist

class RAAC:
    def __init__(
        self,
        environment,
        network,
        policy,
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
        self.policy = policy
        self.gamma = gamma
        self.train_after = train_after
        self.train_freq = train_freq
        self.target_update = target_update
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.max_grad_norm = max_grad_norm
        self.tau = tau

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

        self.verbose = verbose

    def update(self, data):
        s, a, r, s_p, d, len = data

        for idx in reversed(range(len)):
            q = self.q_func(s[idx]).gather(0, a[idx].long())

            with torch.no_grad():
                try:
                    a_p = torch.argmax(self.q_func(s_p[idx]), dim = 1).unsqueeze(1)
                    q_p = self.q_target(s_p[idx]).gather(1, a_p[idx])
                except:
                    q_p = 0

                y = r[idx] + self.gamma * q_p * (1 - d[idx])
                                
            loss = F.mse_loss(q, y)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss

    def update_policy(self, data):
        s_r, a_r, _, _, _, _ = data
        pi_r = self.policy(s_r)
        dist_rollout = Categorical(pi_r)
        log_probs = dist_rollout.log_prob(a_r)
        #print(log_probs)

        with torch.no_grad():
            a_r = a_r.unsqueeze(1).long()
            rel_adv = -self.q_func.get_rel_adv(s_r).gather(1, a_r).squeeze()

        policy_loss = -(log_probs * rel_adv).mean()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

        return policy_loss

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def soft_update(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            a = torch.argmax(self.q_func(torch.from_numpy(s).float()).detach()).numpy()
        else:
            a = self.environment.action_space.sample()

        return a

def main():
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    rollout_buffer = Rollout_Memory()

    dqn_agent = RAAC(
        env,
        Q_duelling(obs_dim, act_dim),
        Policy(obs_dim, act_dim),
        target_update=5,
        verbose=2000,
        learning_rate=5e-4)

    episodes = 0
    s_t = env.reset()

    episodic_rewards = deque(maxlen=50)

    for i in range(1000000):
        a_t = dqn_agent.select_action(s_t)

        """if i > 10000:
            rel_advs = dqn_agent.q_func.get_rel_adv(torch.from_numpy(s_t).float())
            print(f"Relative advantages: {rel_advs}")
            print(f"Action taken: {a_t}")
            policy_out = dqn_agent.policy(torch.from_numpy(s_t).float())
            print(f"Policy saids: {policy_out}")
            print(f"Policy sample: {Categorical(policy_out).sample()}\n")
            #time.sleep(1e-1)"""

        s_tp1, r_t, done, info = env.step(a_t)
        rollout_buffer.push(s_t, a_t, r_t, s_tp1, done)
        s_t = s_tp1

        if i % dqn_agent.verbose == 0 and i > 0:
            avg_r = sum(episodic_rewards) / len(episodic_rewards)
            print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            rollout = rollout_buffer.pop_all()
            loss = dqn_agent.update(rollout)
            if i % dqn_agent.target_update == 0:
                dqn_agent.hard_update()

            #actor_loss = dqn_agent.update_policy(rollout)
            episodes += 1
            episodic_rewards.append(int(info["episode"]["r"]))
            s_t = env.reset()

        

    # Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = dqn_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        s_t = s_tp1 
        if done:
            print(f'Episode Complete, reward = {info["episode"]["r"]}')
            s_t = env.reset()

           

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

