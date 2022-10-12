"""Soft actor critic agent with randomised ensemble double Q-learning: https://arxiv.org/abs/2101.05982"""
import random
import copy
from collections import deque
import sys
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import numpy as np
import gym
from gym.wrappers import RecordEpisodeStatistics

from PytorchContinuousRL.code.utils.memory import Transition
from PytorchContinuousRL.code.utils.models import ddpg_Critic, sac_Actor

class SAC_REDQ():
    """Class for SAC agent"""
    def __init__(self,
        environment,
        actor,
        critic,
        lr=3e-4,
        buffer_size=1000000,
        batch_size=256,
        num_q_nets=10,
        m_sample=2,
        utd_ratio=20,
        alpha=0.2,
        tau=0.005,
        gamma=0.99,
        train_after=100,
        policy_delay=1,
        verbose=500):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_q_nets = num_q_nets
        self.m_sample = m_sample
        self.utd_ratio = utd_ratio
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay
        self.verbose = verbose
        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critics = [critic for _ in range(num_q_nets)]
        self.critics_targets = copy.deepcopy(self.critics)

        self.q_optimizer_list = []
        for q_i in range(num_q_nets):
            self.q_optimizer_list.append(torch.optim.Adam(self.critics[q_i].parameters(), lr=lr))

        self.idx = 0

        # action rescaling
        self.action_scale = int((self.act_high - (self.act_low)) / 2.0)
        self.action_bias = int((self.act_high + (self.act_low)) / 2.0)

    def update(self, batch, i):
        
        # Perform X updates on the critic
        for updates in range(self.utd_ratio):
            s = torch.from_numpy(np.array(batch.s)).float().cuda()
            a = torch.from_numpy(np.array(batch.a)).cuda()
            r = torch.FloatTensor(batch.r).unsqueeze(1).cuda()
            s_p = torch.from_numpy(np.array(batch.s_p)).float().cuda()
            d = torch.IntTensor(batch.d).unsqueeze(1).cuda()

            # Critic update
            with torch.no_grad():
                a_p, log_pi, _ = self.select_action(s_p)
                # REDQ sampling 
                q_targets = [q_target(s_p, a_p) for q_target in np.random.choice(self.critics_targets, self.m_sample, replace=False)]
                q_targets = torch.cat(q_targets, dim=1)
                min_q, _ = torch.min(q_targets, dim=1, keepdim=True)
                min_q_with_log =  min_q - (log_pi * self.alpha)
                y = r + self.gamma * min_q_with_log * (1 - d)
            
            q_vals = [critic(s, a) for critic in self.critics]
            critic_loss = (1/self.num_q_nets) * sum([F.mse_loss(q, y) for q in q_vals])

            # Update critic network parameters
            for q_i in range(self.num_q_nets):
                    self.q_optimizer_list[q_i].zero_grad()

            critic_loss.backward()
            for critic in self.critics:
                #clip_grad_norm_(critic.parameters(), 0.5)
                pass

            for q_i in range(self.num_q_nets):
                    self.q_optimizer_list[q_i].step()

            for critic_target, critic in zip(self.critics_targets, self.critics):
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Delayed Actor update
        if i % self.policy_delay == 0:
            a_pi, log_pi, _ = self.select_action(s)
            avg_q = (torch.cat([critic(s, a_pi) for critic in self.critics], dim=1)).mean(dim=1)
            policy_loss =  -(avg_q - (log_pi * self.alpha)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

        return critic_loss.cpu()

    def select_action(self, s):
        mean, log_std = self.actor(s.cuda())
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
        a = self.select_action(torch.from_numpy(s).float())[0].cpu().detach().numpy()
        return a

    def save_model(self, i):
        torch.save(self.critics.state_dict(), f'models/critic_{i + self.idx}')
        torch.save(self.critics_target.state_dict(), f'models/critic_target_{i + self.idx}')
        torch.save(self.actor.state_dict(), f'models/actor_{i + self.idx}')
        return True

    def load_model(self):
        self.critics.load_state_dict(torch.load(f'models/critic_{self.idx}'))
        self.critics_target.load_state_dict(torch.load(f'models/critic_target_{self.idx}'))
        self.actor.load_state_dict(torch.load(f'models/actor_{self.idx}'))
        return True

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v1'
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    episodic_rewards = deque(maxlen=20)
    episodes = 0
    r_sum = 0

    sac_agent = SAC_REDQ(
        environment=env,
        actor=sac_Actor(4,1).cuda(),
        critic=ddpg_Critic(4,1).cuda(),
        num_q_nets=5,
        lr=7e-4,
        #tau=7.3e-4,
        train_after=10000,
        verbose=500)

    replay_buffer = deque(maxlen=sac_agent.buffer_size)

    s_t = env.reset()

    for i in range(300000):
        if i >= sac_agent.train_after:
            a_t = sac_agent.act(s_t)
        else:
            a_t = env.action_space.sample()
        
        s_tp1, r_t, done, _ = env.step(a_t)
        r_sum += r_t
        replay_buffer.append([s_t, a_t, 0, s_tp1, done])
        s_t = s_tp1

        if len(replay_buffer) >= sac_agent.batch_size and i >= sac_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=sac_agent.batch_size)))
            loss = sac_agent.update(batch, i)

        if i % sac_agent.verbose == 0 and i > 0:
            avg_r = sum(episodic_rewards)/len(episodic_rewards)
            print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")
            
        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            avg_r = sum(episodic_rewards)/len(episodic_rewards)
            if avg_r > 175:
                print("Saving expert model and exiting")
                torch.save(sac_agent.actor.state_dict(), f'sac_redq_expert_{avg_r}')
                break

            r_sum = 0
            s_t = env.reset()

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()