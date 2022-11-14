import sys
import torch
import torch.nn.functional as F
import numpy as np
from memory import ExperienceReplay
from copy import deepcopy

class MADDPG():
    def __init__(
        self,
        environment,
        critic_base=None,
        actor_base=None,
        lr=0.01,
        buffer_size=int(1e6),
        learning_starts=int(5e4), 
        batch_size=1024,
        tau=0.01,
        gamma=0.95,
        update_every=100
    ) -> None:
        # Hyper-parameters
        self.env = environment
        self.lr = lr
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every

        # Multiagent initialisation
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        obs_dims, act_dims, obs_sum, act_sum = self.init_dimensions()

        self.critics = {agent: critic_base(obs_sum, act_sum) for agent in self.agents}
        self.actors = {agent: actor_base(obs_dims[agent], act_dims[agent]) for agent in self.agents}

        self.critic_targets = deepcopy(self.critics)
        self.actor_targets = deepcopy(self.actors)

        self.memory = {agent:ExperienceReplay(self.buffer_size) for agent in self.agents}

        self.c_optim = {agent: torch.optim.Adam(self.critics[agent].parameters(), lr=lr) for agent in self.agents}
        self.a_optim = {agent: torch.optim.Adam(self.actors[agent].parameters(), lr=lr) for agent in self.agents}

        # Exploration noise parameters
        action_max = 1
        action_min = 0
        self.action_mean = (action_max + action_min) / 2
        self.action_std = (action_max - action_min) / 2
        self.epsilon = 1
        self.episolon_period = 10000

        pass

    @property
    def __len__(
        self
    ) -> None:
        return self.memory[self.agents[0]].__len__

    def init_dimensions(
        self
    ) -> list:
        act_dims = {}
        obs_dims = {}
        act_sum = 0
        obs_sum = 0
        for agent in self.agents:
            act_dims[agent] = self.env.action_spaces[agent].n   #shape[0]
            obs_dims[agent] = self.env.observation_spaces[agent].shape[0]
            act_sum += act_dims[agent]
            obs_sum += obs_dims[agent]

        return obs_dims, act_dims, obs_sum, act_sum 
    
    def soft_target_update(
        self
    ) -> None:
        for agent in self.agents:
            for params, target_params in zip(self.actors[agent].parameters(), self.actor_targets[agent].parameters()):
                target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)

            for params, target_params in zip(self.critics[agent].parameters(), self.critic_targets[agent].parameters()):
                target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)

    def update(
        self
    ) -> None:

        s, a, r, s_p, d, a_p = self.multiagent_sample()

        for agent in self.agents:

            ### Critic update
            q_p = self.critic_targets[agent](s_p, a_p)
            y = r[agent] + self.gamma * q_p * (1 - d[agent])

            preds = self.critics[agent](s, a)

            critic_loss = F.mse_loss(preds, y.detach(), reduction='mean') 

            self.c_optim[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.c_optim[agent].step()


            ### Actor update
            logits = self.actors[agent](s[agent])
            a_pi = F.gumbel_softmax(logits).argmax(1)
            a[agent] = torch.zeros([len(a_pi), 5]).scatter(-1, a_pi.unsqueeze(-1), 1)
            actor_loss = -self.critics[agent](s, a).mean()
            actor_loss += 1e-3 * torch.pow(logits, 2).mean()

            self.a_optim[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.a_optim[agent].step()

        return (critic_loss + actor_loss)

    def act(
        self,
        s_t
    ) -> dict:
        a_t = {}
        for agent in self.agents:
            logits = self.actors[agent](torch.from_numpy(s_t[agent]))
            a_t[agent] = F.gumbel_softmax(logits, hard=True).argmax(-1).item()

        return a_t

    def multiagent_store(
        self,
        s_t,
        a_t,
        r_t,
        s_tp1,
        done
    ) -> None:
        for agent in self.agents:
            a_onehot = np.eye(5)[a_t[agent]]
            self.memory[agent].store(s_t[agent], a_onehot, r_t[agent], s_tp1[agent], done[agent])

        pass

    def multiagent_sample(
        self
    ) -> list:
        idxs = np.random.choice(a=self.__len__, size=self.batch_size, replace=False)
        s_b, a_b, r_b, stp1_b, d_b, atp1_b = {}, {}, {}, {}, {}, {}
        for agent in self.agents:
            s, a, r, stp1, d = self.memory[agent].sample(idxs)
            atp1 = F.gumbel_softmax(self.actor_targets[agent](stp1), hard=True).argmax(1).detach()

            s_b[agent] = s
            a_b[agent] = a
            r_b[agent] = r
            stp1_b[agent] = stp1
            d_b[agent] = d
            atp1_b[agent] = torch.zeros([len(atp1), 5]).scatter(-1, atp1.unsqueeze(-1), 1)

        return s_b, a_b, r_b, stp1_b, d_b, atp1_b

