import torch
import numpy as np
import gym
import copy

from ..utils.reward_logger import RewardLogger
from ..utils.models import Critic_quantregression, td3_Actor
from ..utils.memory import ReplayBuffer

class QR_DDPG():
    def __init__(self, 
        env, 
        actor,
        critic,
        buffer_size=1000000, 
        replay_buffer=ReplayBuffer,
        lr=3e-4,
        gamma=0.99,
        batch_size=100, 
        learning_starts=0,
        tau=0.005,
        max_grad_norm=0.5,  # double check value
        N=32,
        kappa=1):

        self.env = env
        self.buffer_size = buffer_size
        self.replay_buffer = replay_buffer(self.buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm

        # distributional variables
        self.N = N
        self.tau_interval = torch.FloatTensor([(n/self.N) for n in range(1, self.N+1)])   # Need to understand!
        self.kappa = kappa

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        #Critic update
        with torch.no_grad():
            a_p = self.actor_target(s_p)
            target_q = self.critic_target(s_p, a_p)
            y = r + self.gamma * target_q * (1 - d)

        q = self.critic(s, a)

        critic_loss = self.quantile_huber_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        policy_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return critic_loss
    
    def quantile_huber_loss(self, q, y):
        u = y - q
        huber_loss = torch.where(u.abs() <= self.kappa, 0.5 * u.pow(2), self.kappa * (u.abs() - 0.5 * self.kappa))
        qh_loss = abs(self.tau_interval - (huber_loss.detach() < 0).float()) * huber_loss / self.kappa
        return qh_loss.sum(-1).mean()

    def soft_target_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def select_action(self, s):
        a = self.actor(torch.from_numpy(s).float()).detach().numpy()
        return a

    def train(self, train_steps, report_freq):
        logger = RewardLogger(report_freq=report_freq)

        # Training loop
        s_t = self.env.reset()
        for i in range(train_steps):
            if i > self.learning_starts:
                a_t = self.select_action(s_t)
            else:
                a_t = self.env.action_space.sample()
                
            s_tp1, r_t, done, _ = self.env.step(a_t)
            logger.log_step(r_t)
            self.replay_buffer.append(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

            if len(self.replay_buffer) >= self.batch_size and i >= self.learning_starts:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.update(batch)
                self.soft_target_update()

            if done:
                logger.end_of_eps()
                s_t = self.env.reset()

        return logger.best

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = gym.make(env_name)
    
    qrddpg_agent = QR_DDPG(
        env,
        td3_Actor(env.observation_space.shape[0], env.action_space.shape[0]),
        Critic_quantregression(env.observation_space.shape[0], env.action_space.shape[0]), 
        lr=7e-4,
        tau=0.01,
        learning_starts=10000,
        gamma=0.98,)

    qrddpg_agent.train(train_steps=100000, report_freq=1000)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

