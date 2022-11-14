import yaml
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from memory import ExperienceReplay
from copy import deepcopy
from ma_models import local_Actor, central_twinq_Critic
from reward_logger import RewardLogger

# environments
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.mpe import simple_v2, simple_tag_v2, simple_spread_v2

class MATD3():
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
        update_every=100,
        action_max=1,
        action_min=0
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
        self.action_max = action_max
        self.action_min = action_min

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
            act_dims[agent] = (self.env.action_space(agent).shape[0] - 1)
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
        self,
        i
    ) -> None:
        s, a, r, s_p, d, a_p = self.multiagent_sample()

        for agent in self.agents:            
            ### Critic update
            q_p1, q_p2 = self.critic_targets[agent](s_p, a_p)
            q_p = torch.min(q_p1, q_p2)
            y = r[agent] + self.gamma * q_p * (1 - d[agent])

            preds1, preds2 = self.critics[agent](s, a)

            critic_loss = F.mse_loss(preds1, y.detach()) + F.mse_loss(preds2, y.detach())
            
            self.c_optim[agent].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.c_optim[agent].step()
                
            ### Delayed actor update
            if i % (2*self.update_every) == 0:
                a_temp = a[agent]   
                # awkward, should look into (long story short, the output of the actor keeps its gradient which then interupts the next iteration)
                
                a[agent] = self.actors[agent](s[agent])
                actor_loss = -self.critics[agent].q1(s, a).mean()
                a[agent] = a_temp

                self.a_optim[agent].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
                self.a_optim[agent].step()

                self.soft_target_update()

    def act(
        self,
        s_t
    ) -> dict:
        """"
        TD3 style noise action for policy smoothing and exploration
        """
        a_t = {}
        for agent in self.agents:
            action = self.actors[agent](torch.from_numpy(s_t[agent]))
            noise = (torch.normal(torch.zeros_like(action), 0.1)).clamp(-0.2, 0.2)
            a_t[agent] = (action + noise).clamp(self.action_min, self.action_max).detach().numpy()
            # print(a_t[agent])

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

def convert_actions(a_t, env):
    new_a_t = {}
    for agent in env.possible_agents:
        new_a_t[agent] = np.insert(a_t[agent], 0, 0)
        
    return new_a_t

"""
Converging env's and the params:
Simple_v2, max_cycles=50:
-lr=0.001,
-HIDDEN_DIMS=128,
~4000 episodes

# at N=2 for simple_spread, only one agent learnt to go for the landmark, the other agent learnt to remain stationary?,
# not good behaviour, try fix

Simple_spread_v2, local_ratio=, N=:

"""

def main():
    with open("configs/spread_config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # env = simple_v2.parallel_env(continuous_actions=True)
    env = simple_spread_v2.parallel_env(
        N=cfg['env']['n_agents'],
        local_ratio=cfg['env']['ratio'],
        max_cycles=cfg['env']['cycles'],
        continuous_actions=True)

    logger  = RewardLogger(env.possible_agents, episode_avg=100, update_freq=50)

    maddpg_agent = MATD3(
        environment=env,
        actor_base=local_Actor,
        critic_base=central_twinq_Critic,
        lr=cfg['hparams']['lr'],
        batch_size=100) #matd3 paper used batchsize of 100 and lr of 0.001

    i = 0

    for _ in range(cfg['train_episodes']):
        s_t = env.reset()
        while(env.agents):
            i += 1
            if i == maddpg_agent.learning_starts:
                print('/n### Beginning training ###')

            if i >= maddpg_agent.learning_starts:
                a_t = maddpg_agent.act(s_t)
                
            else:
                a_t = {agent: env.action_space(agent).sample()[1:] for agent in env.possible_agents}
                
            s_tp1, r_t, done, trun, _ = env.step(convert_actions(a_t, env))
            logger.log_step(r_t)

            maddpg_agent.multiagent_store(s_t, a_t, r_t, s_tp1, done)
            s_t = s_tp1

            if len(maddpg_agent) > maddpg_agent.batch_size and i % maddpg_agent.update_every == 0 and i > maddpg_agent.learning_starts:
                maddpg_agent.update(i)

        logger.end_of_eps()

    logger.plot_training()

    ### Render trained agents ###
    while True:
        s_t = env.reset()
        while(env.agents):
            a_t = maddpg_agent.act(s_t)
            s_tp1, r_t, done, _, _ = env.step(convert_actions(a_t, env))
            print(r_t)
            env.render()
            time.sleep(0.2)

if __name__ == "__main__":
    main()
    