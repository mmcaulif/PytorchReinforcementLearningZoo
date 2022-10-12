"""Soft actor critic agent"""
import random
import copy
from collections import deque
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import numpy as np
import gym
from gym.wrappers import RecordEpisodeStatistics

from PytorchContinuousRL.code.utils.models import twinq_Critic, sac_Actor
from PytorchContinuousRL.code.utils.memory import Transition

class SAC():
    """Class for SAC agent"""
    def __init__(self,
        environment,
        actor,
        critic,
        lr=3e-4,
        buffer_size=1000000,
        batch_size=256,
        alpha=0.2,
        tau=0.005,
        gamma=0.99,
        train_after=100,
        policy_delay=1,
        verbose=500,
        idx=0):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
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

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.idx = idx

        # action rescaling
        #print(self.act_high, self.act_low)
        self.action_scale = int((self.act_high - (self.act_low)) / 2.0)
        self.action_bias = int((self.act_high + (self.act_low)) / 2.0)
        #print(self.action_scale, self.action_bias)

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).float().cuda()
        a = torch.from_numpy(np.array(batch.a)).cuda()
        r = torch.FloatTensor(batch.r).unsqueeze(1).cuda()
        s_p = torch.from_numpy(np.array(batch.s_p)).float().cuda()
        d = torch.IntTensor(batch.d).unsqueeze(1).cuda()

        #Critic update
        with torch.no_grad():
            a_p, log_pi, _ = self.select_action(s_p)
            target_q1, target_q2 = self.critic_target(s_p, a_p)
            target_q = torch.min(target_q1, target_q2) - (log_pi * self.alpha)
            y = r + self.gamma * target_q * (1 - d)

        q1, q2 = self.critic(s, a)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            a_pi, log_pi, _ = self.select_action(s)
            policy_loss = -(self.critic.q1_forward(s, a_pi) - (log_pi * self.alpha)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

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
        torch.save(self.critic.state_dict(), f'models/critic_{i + self.idx}')
        torch.save(self.critic_target.state_dict(), f'models/critic_target_{i + self.idx}')
        torch.save(self.actor.state_dict(), f'models/actor_{i + self.idx}')
        return True

    def load_model(self):
        self.critic.load_state_dict(torch.load(f'models/critic_{self.idx}'))
        self.critic_target.load_state_dict(torch.load(f'models/critic_target_{self.idx}'))
        self.actor.load_state_dict(torch.load(f'models/actor_{self.idx}'))
        return True

def main():
    #env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    episodic_rewards = deque(maxlen=20)
    episodes = 0
    r_sum = 0

    sac_agent = SAC(environment=env,    #taken from sb3 zoo
        actor=sac_Actor(2,1).cuda(),
        critic=twinq_Critic(2,1).cuda(),
        lr=1e-3,
        tau=7.3e-4,
        train_after=1000,
        verbose=2000)

    replay_buffer = deque(maxlen=sac_agent.buffer_size)

    s_t = env.reset()

    for i in range(300000):
        if i >= sac_agent.train_after:
            a_t = sac_agent.act(s_t)
        else:
            a_t = env.action_space.sample()
        
        s_tp1, r_t, done, _ = env.step(a_t)
        r_sum += r_t
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
        s_t = s_tp1

        if len(replay_buffer) >= sac_agent.batch_size and i >= sac_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=sac_agent.batch_size)))
            loss = sac_agent.update(batch, i)

            if i % sac_agent.verbose == 0:
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_t = env.reset()

    #Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = sac_agent.act(s_t)
        s_tp1, r_t, done, _ = env.step(a_t)
        if done:
            s_tp1 = env.reset()

        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()