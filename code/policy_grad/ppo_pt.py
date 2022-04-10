import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from utils.models import PPO_model
from utils.memory import Rollout_Memory

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py  shows off the ratios well
class PPO:
    def __init__(
        self,
        network,
        gamma=0.99,
        lmbda=0.95,
        ent_coeff=0.1,
        n_steps=2048,
        batch_size=64,
        verbose=20,
        learning_rate=2.5e-4,
        clip_range=0.2,
        max_grad_norm=0.5
    ):
        self.network = network
        self.gamma = gamma
        self.lmbda = lmbda
        self.ent_coeff = ent_coeff
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)   #ppo optimiser
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm

    def calc_gae(self, s_r, r_r, d_r, final_r): #Still need to understand!
        gae_returns = torch.zeros_like(r_r)
        gae = 0
        for t in reversed(range(len(r_r))):
            if t == len(r_r) - 1: 
                v_tp1 = final_r
            else: 
                v_tp1 = self.network(s_r[t+1])[0].squeeze(-1)

            v_t = self.network(s_r[t])[0].squeeze(-1)
            y = r_r[t] + v_tp1 * self.gamma * (1 - d_r[t])
            td = y - v_t
            gae = td + self.gamma * self.lmbda * gae * (1 - d_r[t])

            gae_returns[t] = gae + v_t

        return gae_returns

    def update(self, data, r_traj):
        s_r, a_r, r_r, pi_r, d_r, len = data
        
        Q_rollout = self.calc_gae(s_r, r_r, d_r, r_traj).detach()

        V_rollout = self.network(s_r.float())[0]

        old_probs = Categorical(pi_r).log_prob(a_r).detach()

        for iter in range(0, len, self.batch_size):  
            iter_end = iter + self.batch_size

            #new outputs
            new_v, new_pi = self.network(s_r[iter:iter_end])

            #critic
            critic_loss = F.mse_loss(Q_rollout[iter:iter_end], new_v.squeeze(-1))

            #actor
            new_dist = Categorical(new_pi)
            log_probs = new_dist.log_prob(a_r[iter:iter_end])
            entropy = new_dist.entropy().mean()   #ppo entropy loss
            
            adv = Q_rollout[iter:iter_end] - V_rollout[iter:iter_end].detach()

            ratio = torch.exp(log_probs - old_probs[iter:iter_end])   #ppo policy loss function
            clipped_ratio = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
            actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

            loss = actor_loss + (critic_loss * 0.5) - (entropy * self.ent_coeff)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm) #ppo gradient clipping
            self.optimizer.step()

        return loss
        
    def select_action(self, s):
        a_policy = self.network(torch.from_numpy(s).float())[1]
        a = int(torch.multinomial(a_policy, 1).detach())
        return a, a_policy

def main():
    env_name = "CartPole-v1"
    #env_name = "LunarLander-v2"
    num_envs = 1
    #env = RecordEpisodeStatistics(gym.vector.make(env_name, num_envs=num_envs))
    env = RecordEpisodeStatistics(gym.make(env_name))
    obs_dim = 4
    act_dim = 2
    avg_r = deque(maxlen=50)

    buffer = Rollout_Memory()
    ppo_agent = PPO(PPO_model(obs_dim, act_dim), 0.98, 0.8, 0.0, 1024, 4)

    s_t = env.reset()
    for i in range(1, 5000):
        r_trajectory = 0
        while buffer.qty < ppo_agent.n_steps:
            a_t, a_pi = ppo_agent.select_action(s_t)
            s_tp1, r, d, info = env.step(a_t)
            #[buffer.push(s_t[i], a_t[i], r[i], a_pi[i], d[i]) for i in range(num_envs)]
            buffer.push(s_t, a_t, r, a_pi, d)

            s_t = s_tp1
            if d:
                s_t = env.reset()
                avg_r.append(int(info["episode"]["r"]))
                if i % ppo_agent.verbose == 0:
                    print(f'Episode: {i} | Average reward: {sum(avg_r)/len(avg_r)} | [{len(avg_r)}]')
                break

        if not d:
            r_trajectory = ppo_agent.network(torch.from_numpy(s_tp1).float())[0]
        
        data = buffer.pop_all()
        #print(data[:-1], r_trajectory, '\n')
        ppo_agent.update(data, r_trajectory)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

