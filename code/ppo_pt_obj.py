import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
#from utils.models import PPO_model
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
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=1e-5)   #ppo optimiser

    def select_action(self, s):
        a_policy = self.network(torch.from_numpy(s).float())[1]
        a = int(torch.multinomial(a_policy, 1).detach())
        return a, a_policy

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

    def update(self, s_r, a_r, r_r, pi_r, d_r, len, r_traj):
        Q_rollout = self.calc_gae(s_r, r_r, d_r, r_traj).detach()

        V_rollout = self.network(s_r.float())[0]

        old_probs = Categorical(pi_r).log_prob(a_r).detach()

        for iter in range(len):  
            #new outputs
            new_v, new_pi = self.network(s_r[iter])

            #critic
            critic_loss = F.mse_loss(Q_rollout[iter], new_v.squeeze(-1))

            #actor
            new_dist = Categorical(new_pi)
            log_probs = new_dist.log_prob(a_r[iter])
            entropy = new_dist.entropy().mean()   #ppo entropy loss
            
            adv = Q_rollout[iter] - V_rollout[iter].detach()

            #if len(adv) > 1: adv = (adv - adv.mean())/(adv.std() + 1e-8) #ppo minibatch advantage normalisations

            ratio = torch.exp(log_probs - old_probs[iter])   #ppo policy loss function
            clipped_ratio = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
            actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

            loss = actor_loss + (critic_loss * 0.5) - (entropy * self.ent_coeff)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm) #ppo gradient clipping
            self.optimizer.step()

            return loss

class PPO_model(torch.nn.Module):
    def __init__(self, input_size, action_size):
        super(PPO_model, self).__init__()
        
        self.critic = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, 1)
        )
        self.actor = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, s):
        v = self.critic(s)
        pi = self.actor(s)
        return v, pi

def main():
    env_name = "CartPole-v1"
    env = RecordEpisodeStatistics(gym.make(env_name))
    avg_r = deque(maxlen=50)
    count = 0

    buffer = Rollout_Memory()
    net = PPO_model(4, 2)
    ppo_agent = PPO(net)

    s_t = env.reset()
    for i in range(50000):

        while buffer.qty <= ppo_agent.n_steps:
            a_t, a_pi = ppo_agent.select_action(s_t)
            s_tp1, r, d, info = env.step(a_t)
            buffer.push(s_t, a_t, r, a_pi, d)
            s_t = s_tp1
            if d:
                s_t = env.reset()
                count += 1
                avg_r.append(int(info["episode"]["r"]))
                if count % ppo_agent.verbose == 0:
                    print(f'Episode: {count} | Average reward: {sum(avg_r)/len(avg_r)} | [{len(avg_r)}]')
                break

        if d:
            r_trajectory = 0
        else:
            r_trajectory = ppo_agent.network(torch.from_numpy(s_tp1).float())[0]
        
        #s_rollout, a_rollout, r_rollout, pi_rollout, d_rollout, len_rollout = buffer.pop_all()
        #ppo_agent.update(s_rollout, a_rollout, r_rollout, pi_rollout, d_rollout, len_rollout, r_trajectory)
        total_loss = ppo_agent.update(*buffer.pop_all(), r_trajectory)
        #print(total_loss)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

