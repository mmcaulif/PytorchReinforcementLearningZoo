from collections import deque
import sys
import torch
import torch.nn.functional as F
import numpy as np
from memory import ExperienceReplay
from copy import deepcopy
from ma_models import local_Actor, central_Critic
from pettingzoo.sisl import multiwalker_v9

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

        # Exploration noise parameters for MATD3
        action_max = 1
        action_min = 0
        self.action_mean = (action_max + action_min) / 2
        self.action_std = (action_max - action_min) / 2
        self.epsilon = 1
        self.epsilon_period = 10000

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
            act_dims[agent] = self.env.action_space(agent).shape[0]
            obs_dims[agent] = self.env.observation_space(agent).shape[0]
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

            critic_loss = F.mse_loss(preds, y.detach())
            
            self.c_optim[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.c_optim[agent].step()


            ### Actor update
            a_temp = a[agent]   
            # awkward, should look into (long story short, the output of the actor keeps its gradient which then interupts the next iteration)
            
            a[agent] = self.actors[agent](s[agent])
            actor_loss = -self.critics[agent](s, a).mean()
            a[agent] = a_temp

            self.a_optim[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.a_optim[agent].step()

    def act(
        self,
        s_t
    ) -> dict:
        a_t = {agent: self.actors[agent](torch.from_numpy(s_t[agent])).detach().numpy() for agent in self.agents}
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
        idxs = np.random.choice(a=self.__len__, size=self.batch_size, replace=False)
        s_b, a_b, r_b, stp1_b, d_b, atp1_b = {}, {}, {}, {}, {}, {}
        for agent in self.agents:
            s, a, r, stp1, d = self.memory[agent].sample(idxs)

            s_b[agent] = s
            a_b[agent] = a
            r_b[agent] = r
            stp1_b[agent] = stp1
            d_b[agent] = d
            atp1_b[agent] = self.actor_targets[agent](stp1).detach()

        return s_b, a_b, r_b, stp1_b, d_b, atp1_b

def main():
    env = multiwalker_v9.parallel_env(shared_reward=False)
    print(env.possible_agents)

    maddpg_agent = MADDPG(
        environment=env,
        actor_base=local_Actor,
        critic_base=central_Critic)

    n_episodes = 4000

    i = 0
    r_sum = [0] * len(env.possible_agents)
    r_avg = deque(maxlen=50)
    for eps in range(n_episodes):

        s_t = env.reset()
        while(env.agents):
            i += 1
            if i == maddpg_agent.learning_starts:
                print('### Beginning training ###')

            if i > maddpg_agent.learning_starts:
                a_t = maddpg_agent.act(s_t)
            else:
                a_t = {agent: env.action_space(agent).sample() for agent in env.possible_agents}            

            s_tp1, r_t, done, trun, _ = env.step(a_t)

            r_sum = [x + y for x, y in zip(r_sum, r_t.values())]

            maddpg_agent.multiagent_store(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

            if maddpg_agent.__len__ > maddpg_agent.batch_size and i % maddpg_agent.update_every == 0 and i > maddpg_agent.learning_starts:
                maddpg_agent.update()
                maddpg_agent.soft_target_update()

        r_avg.append(r_sum)
        r_sum = [0] * len(env.possible_agents)

        if eps % 25 == 0 and eps > 0:
            agent_performance_list = [round(sum(x) / len(x), 3) for x in zip(*r_avg)]
            agent_performance_dict = {}
            for j, agent in enumerate(env.possible_agents):
                agent_performance_dict[agent] = agent_performance_list[j]
            print(f"EPISODE {eps} DONE, AVERAGE REWARDS: {agent_performance_dict}")


if __name__ == "__main__":
    main()

