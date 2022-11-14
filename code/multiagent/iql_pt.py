import torch
import torch.nn.functional as F
import numpy as np
#from .memory import ExperienceReplay
from copy import deepcopy

from ..utils.memory import ReplayBuffer as ExperienceReplay
from ..utils.reward_logger import MultiagentRewardLogger

class IQL():
    def __init__(
        self,
        environment,
        network=None,
        lr=0.01,
        buffer_size=int(1e6),
        learning_starts=int(5e4), 
        batch_size=1024,
        tau=0.01,
        gamma=0.95,
        train_freq=100,
        target_update=1000
    ) -> None:
        # Hyper-parameters
        self.env = environment
        self.lr = lr
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update = target_update

        # Multiagent initialisation
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        obs_dims, act_dims, _, _ = self.init_dimensions()

        self.q_funcs = {agent: network(obs_dims[agent], act_dims[agent]) for agent in self.agents}

        self.q_targets = deepcopy(self.q_funcs)

        self.memory = {agent:ExperienceReplay(self.buffer_size) for agent in self.agents}

        self.optim = {agent: torch.optim.Adam(self.q_funcs[agent].parameters(), lr=lr) for agent in self.agents}

    def __len__(
        self
    ) -> int:
        return len(self.memory[self.agents[0]])

    def init_dimensions(
        self
    ) -> list:
        act_dims = {}
        obs_dims = {}
        act_sum = 0
        obs_sum = 0
        for agent in self.agents:
            act_dims[agent] = (self.env.action_space(agent).n)
            obs_dims[agent] = self.env.observation_space(agent).shape[0]
            act_sum += act_dims[agent]
            obs_sum += obs_dims[agent]

        return obs_dims, act_dims, obs_sum, act_sum 
    
    def hard_target_update(
        self
    ) -> None:
        self.q_targets = deepcopy(self.q_funcs)
        pass

    def update(
        self,
    ) -> None:
        s, a, r, s_p, d = self.multiagent_sample()

        for agent in self.agents:

            q = self.q_funcs[agent](s[agent]).gather(1, a[agent].unsqueeze(-1).long()).squeeze()

            with torch.no_grad():
                q_p = self.q_targets[agent](s_p[agent]).amax()
                y = r[agent] + self.gamma * q_p * (1 - d[agent])

            critic_loss = F.mse_loss(q, y)
            
            self.optim[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_funcs[agent].parameters(), 0.5)
            self.optim[agent].step()

        pass

    def act(
        self,
        s_t
    ) -> dict:
        a_t = {}
        for agent in self.agents:
            q_values = self.q_funcs[agent](torch.from_numpy(s_t[agent]))
            a_pi = q_values.argmax()
            temp = np.zeros(5)
            temp[a_pi] = 1
            a_t[agent] = a_pi.item()

        return a_t

    def multiagent_store(
        self,
        s_t,
        a_t,
        r_t,
        s_tp1,
        done
    ) -> None:
        for agent, memory in self.memory.items():
            memory.store(s_t[agent], a_t[agent], r_t[agent], s_tp1[agent], done[agent])


    def multiagent_sample(
        self
    ) -> list:
        idxs = np.random.choice(a=self.__len__(), size=self.batch_size, replace=False)
        s_b, a_b, r_b, stp1_b, d_b = {}, {}, {}, {}, {}
        for agent in self.agents:
            s, a, r, stp1, d = self.memory[agent].sample(idxs)

            s_b[agent] = s
            a_b[agent] = a
            r_b[agent] = r
            stp1_b[agent] = stp1
            d_b[agent] = d

        return s_b, a_b, r_b, stp1_b, d_b

    def train(
        self,
        train_steps,
        report_freq=20
    ) -> float:
        logger = MultiagentRewardLogger(
            agents=self.agents,
            episode_avg=40,
            report_freq=report_freq)

        # Training loop
        s_t = self.env.reset()
        for i in range(train_steps):
            if i > self.learning_starts:
                a_t = self.act(s_t)
            else:
                a_t = {agent: self.env.action_space(agent).sample() for agent in self.agents}
                
            s_tp1, r_t, done, trun, _ = self.env.step(a_t)
            logger.log_step(r_t)
            self.multiagent_store(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

            if len(self) >= self.batch_size and i >= self.learning_starts:
                
                if i % self.train_freq == 0:
                    self.update()

                if i % self.target_update == 0:
                    self.hard_target_update()

            if any(done.values()) or any(trun.values()):
                logger.end_of_eps()
                s_t = self.env.reset()

        return logger.stats()