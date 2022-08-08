import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym
from gym import spaces
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from PytorchContinuousRL.code.utils.models import PPO_model, PPO_cont_model, A2C_Model
from PytorchContinuousRL.code.utils.memory import Rollout_Memory

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
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        k_epochs=10
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
        self.k_epochs = k_epochs

        #KL divergence
        self.use_kl = True
        self.target_kl = 0.03

    def calc_gae(self, s_r, r_r, d_r, final_r): #Still need to understand!
        gae_returns = torch.zeros_like(r_r)
        advantages = torch.zeros_like(r_r)
        v_t = torch.zeros_like(r_r)
        gae = 0
        for t in reversed(range(len(r_r))):
            if t == len(r_r) - 1: 
                v_tp1 = final_r
            else: 
                v_tp1 = self.network.critic(s_r[t+1]).squeeze(-1)

            v_t[t] = self.network.critic(s_r[t]).squeeze(-1)
            y = r_r[t] + v_tp1 * self.gamma * (1 - d_r[t])
            td = y - v_t[t]
            advantages[t] = gae = td + self.gamma * self.lmbda * gae * (1 - d_r[t])

        gae_returns = advantages + v_t

        return gae_returns.detach(), advantages.detach()

    def update(self, data, r_traj):
        s_r, a_r, r_r, old_probs, d_r, len = data
        
        Q_rollout, adv_rollout = self.calc_gae(s_r, r_r, d_r, r_traj)

        indxs = np.arange(len)

        for _ in range(self.k_epochs):
            np.random.shuffle(indxs)

            for iter in range(0, len, self.batch_size):  
                iter_end = iter + self.batch_size
                mb_indxs = indxs[iter:iter_end]

                #new outputs
                new_v, new_pi = self.network(s_r[mb_indxs])

                #critic
                critic_loss = F.mse_loss(Q_rollout[mb_indxs], new_v.squeeze(-1))

                #actor
                #new_dist = self.network.get_dist(s_r[mb_indxs])
                mu, sigma = new_pi
                new_dist = Normal(mu, sigma)
                
                log_probs = new_dist.log_prob(a_r[mb_indxs])
                entropy = new_dist.entropy().mean()   #ppo entropy loss
                
                adv = adv_rollout[mb_indxs]

                log_ratio = log_probs - old_probs[mb_indxs].squeeze()
                ratio = log_ratio.exp()   #ppo policy loss function

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                clipped_ratio = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
                actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

                loss = actor_loss + (critic_loss * 0.5) - (entropy * self.ent_coeff)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm) #ppo gradient clipping
                self.optimizer.step()

            if self.use_kl:
                if approx_kl > self.target_kl:
                    #stopping early
                    return loss

        return loss
        
    def select_action(self, s):
        a_policy = self.network.get_dist(torch.from_numpy(s).float())
        a = a_policy.sample().clamp(-1, 1)  #for cartpole continuous
        a_log_prob = a_policy.log_prob(a)
        return a.detach().numpy(), a_log_prob.detach()

def main():
    #env_name = "CartPole-v1"
    #env_name = "LunarLander-v2"
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = RecordEpisodeStatistics(gym.make(env_name))
    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.n
    except:
        act_dim = env.action_space.shape[0]

    avg_r = deque(maxlen=50)

    if isinstance(env.action_space, spaces.Discrete):
        print("Discrete action space")
        model = PPO_model(obs_dim, act_dim, net_size=256)
    else:
        print("Continuous action space")
        model = PPO_cont_model(obs_dim, act_dim, net_size=256)

    model = A2C_Model(obs_dim, act_dim)

    buffer = Rollout_Memory()
    ppo_agent = PPO(
        model,
        n_steps=8,
        batch_size=64)

    s_t = env.reset()
    for i in range(1, 5000):
        r_trajectory = 0
        while buffer.qty < ppo_agent.n_steps:
            a_t, a_logpi = ppo_agent.select_action(s_t)
            s_tp1, r, d, info = env.step(a_t)
            buffer.push(s_t, a_t, r, a_logpi, d)

            s_t = s_tp1
            if d:
                s_t = env.reset()
                avg_r.append(int(info["episode"]["r"]))
                if i % ppo_agent.verbose == 0:
                    print(f'Episode: {i} | Average reward: {sum(avg_r)/len(avg_r)} | [{len(avg_r)}]')
                break

        if not d:
            r_trajectory = ppo_agent.network.critic(torch.from_numpy(s_tp1).float()).detach()
        
        data = buffer.pop_all()
        ppo_agent.update(data, r_trajectory)

if __name__ == "__main__":
    main()

