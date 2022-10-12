import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from PytorchContinuousRL.code.utils.models import A2C_Model
from PytorchContinuousRL.code.utils.memory import Rollout_Memory

class A2C():
    def __init__(
        self,
        network,
        gamma=0.99,
        ent_coeff=0.1,
        critic_coeff=0.5,
        n_steps=8,
        verbose=20,
        learning_rate=7e-4,
        max_grad_norm=0.5
    ):
        self.network = network
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.critic_coeff = critic_coeff
        self.n_steps = n_steps
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.max_grad_norm = max_grad_norm

    def select_action(self, s):
        dist = self.network.get_dist(torch.from_numpy(s).float())
        a_t = dist.sample().clamp(-1, 1)
        return a_t.numpy(), dist.log_prob(a_t)

    def calc_returns(self, r_rollout, final_r):
        discounted_r = torch.zeros_like(r_rollout)
        for t in reversed(range(len(r_rollout))):
            final_r = final_r *  self.gamma + r_rollout[t]
            discounted_r[t] = final_r
        return discounted_r

    def update(self, batch, r_traj):
        s_rollout, a_rollout, r_rollout, _, _, _ = batch
    
        Q = self.calc_returns(r_rollout, r_traj)
    
        V = self.network.critic(s_rollout.float())
        critic_loss = F.mse_loss(Q, V.squeeze(-1))    
        
        dist_rollout = self.network.get_dist(s_rollout.float())
        log_probs = dist_rollout.log_prob(a_rollout)
        entropy = dist_rollout.entropy().mean()
        adv = Q - V.detach()

        actor_loss = -(log_probs.sum() * adv.detach()).mean()
        loss = actor_loss + (critic_loss * self.critic_coeff) - (entropy * self.ent_coeff)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = RecordEpisodeStatistics(gym.make(env_name))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    a2c_agent = A2C(A2C_Model(obs_dim, act_dim), gamma=0.9, ent_coeff=0.01)
    buffer = Rollout_Memory()

    avg_r = deque(maxlen=40)
    count = 0

    s_t = env.reset()
    for i in range(50000):
        r_trajectory = 0
        while buffer.qty <= a2c_agent.n_steps:
            a_t, a_log = a2c_agent.select_action(s_t)
            s_tp1, r, d, info = env.step(a_t)
            buffer.push(s_t, a_t, r, a_log, d)
            s_t = s_tp1
            if d:
                s_t = env.reset()
                count += 1
                avg_r.append(int(info["episode"]["r"]))
                if count % 20 == 0:
                    print(f'Episode: {count} | Average reward: {sum(avg_r)/len(avg_r)} | Rollouts: {i} | [{len(avg_r)}]')
                break

        if not d:
            r_trajectory = a2c_agent.network.critic(torch.from_numpy(s_tp1).float())  

        rollout = buffer.pop_all()
        a2c_agent.update(rollout, r_trajectory)

    

