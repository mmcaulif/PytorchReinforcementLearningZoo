import torch
import numpy as np
import gym
import copy

from ..utils.reward_logger import RewardLogger
from ..utils.models import Q_quantregression
from ..utils.memory import ReplayBuffer

class QR_DQN:
    def __init__(
        self,
        env,
        network,
        replay_buffer=ReplayBuffer,
        gamma=0.99,
        learning_starts=50000,
        train_freq=4,
        target_update=1000,
        batch_size=64,
        learning_rate=1e-4,
        max_grad_norm=10,
        N=32,
        kappa=1
    ):
        self.env = env
        self.replay_buffer = replay_buffer(100000)
        self.q_func = network
        self.q_target = copy.deepcopy(self.q_func)
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update = target_update
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm

        # distributional variables
        self.N = N
        self.tau_interval = torch.FloatTensor([(n/self.N) for n in range(1, self.N+1)])   # Need to understand!
        self.kappa = kappa

        self.EPS_END = 0.05
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a)).unsqueeze(-1).type(torch.int64)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        q = self.q_func(s).squeeze().gather(1, a.unsqueeze(-1).expand(self.batch_size, 1, self.N))

        with torch.no_grad():
            q_dist = self.q_target(s_p).squeeze()
            a_p = q_dist.mean(-1).argmax(-1, keepdim=True).unsqueeze(-1)
            q_p = q_dist.gather(1, a_p.expand(self.batch_size, 1, self.N))
            y = r.unsqueeze(-1) + self.gamma * q_p * (1 - d.unsqueeze(-1))

        loss = self.quantile_huber_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    def quantile_huber_loss(self, q, y):
        u = y - q
        huber_loss = torch.where(u.abs() <= self.kappa, 0.5 * u.pow(2), self.kappa * (u.abs() - 0.5 * self.kappa))
        qh_loss = abs(self.tau_interval - (huber_loss.detach() < 0).float()) * huber_loss / self.kappa
        return qh_loss.sum(2).mean()

    def hard_update(self):
        self.q_target = copy.deepcopy(self.q_func)

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            a_mean = self.q_func(torch.from_numpy(s).float()).mean(dim=-1).detach()
            a = torch.argmax(a_mean).numpy()
        else:
            a = self.env.action_space.sample()

        return a

    def train(self, train_steps, report_freq=20):
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
                
                if i % self.train_freq == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.update(batch)

                if i % self.target_update == 0:
                    self.hard_update()

            if done:
                logger.end_of_eps()
                s_t = self.env.reset()

        return logger.stats()

def main():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    c51_agent = QR_DQN(
        env,
        Q_quantregression,
        learning_starts=250,
        batch_size=256,
        target_update=300,
        learning_rate=2.3e-3,)

    c51_agent.train(train_steps=100000, r_avg_len=20, report_freq=2000)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

