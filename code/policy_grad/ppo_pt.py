import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from PytorchContinuousRL.code.utils.models import A2C_Model
from PytorchContinuousRL.code.utils.memory import Rollout_Memory

class PPO():
    def __init__(
        self,
        network,
        gamma=0.99,
        lmbda=0.95,
        ent_coeff=0.1,
        critic_coeff=0.5,
        n_steps=2048,
        batch_size=64,
        verbose=500,
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        k_epochs=10,
        target_kl=None
    ):
        self.network = network.cuda()
        self.gamma = gamma
        self.lmbda = lmbda
        self.ent_coeff = ent_coeff
        self.critic_coeff = critic_coeff
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.k_epochs = k_epochs
        self.target_kl = target_kl

    def select_action(self, s):
        dist = self.network.get_dist(torch.from_numpy(s).float().cuda())
        a_t = dist.sample()
        return a_t.cpu().numpy(), dist.log_prob(a_t).cpu()

    def calc_returns(self, r_rollout, final_r):
        discounted_r = torch.zeros_like(r_rollout)
        for t in reversed(range(len(r_rollout))):
            final_r = final_r *  self.gamma + r_rollout[t]
            discounted_r[t] = final_r
        return discounted_r.detach()

    def calc_gae(self, s_r, r_r, d_r, final_r): #Still need to understand!
        gae_returns = torch.zeros_like(r_r)
        advantages = torch.zeros_like(r_r)
        v_t = torch.zeros_like(r_r)
        gae = 0
        for t in reversed(range(len(r_r))):
            if t == len(r_r) - 1: 
                v_tp1 = final_r
            else: 
                v_tp1 = self.network.critic(s_r[t+1].cuda()).squeeze(-1)

            v_t[t] = self.network.critic(s_r[t].cuda()).squeeze(-1)
            y = r_r[t] + v_tp1 * self.gamma * (1 - d_r[t])
            td = y - v_t[t]
            advantages[t] = gae = td + self.gamma * self.lmbda * gae * (1 - d_r[t])

        gae_returns = advantages + v_t

        return gae_returns.detach(), advantages.detach()

    def update(self, batch, r_traj):
        s_rollout, a_rollout, r_rollout, old_probs, dones, len = batch
        s_rollout = s_rollout.cuda()
        a_rollout = a_rollout.cuda() 
        r_rollout = r_rollout.cuda() 
        old_probs = old_probs.cuda() 
        dones = dones.cuda()
        
        indxs = np.arange(len)
        
        Q_rollout, adv_rollout = self.calc_gae(s_rollout, r_rollout, dones, r_traj)

        for _ in range(self.k_epochs):
            np.random.shuffle(indxs)

            for iter in range(0, len, self.batch_size):  
                iter_end = iter + self.batch_size
                idx = indxs[iter:iter_end]

                V = self.network.critic(s_rollout[idx].float().cuda())
                critic_loss = F.mse_loss(Q_rollout[idx], V.squeeze(-1))    
                
                dist_rollout = self.network.get_dist(s_rollout[idx].float().cuda())
                new_probs = dist_rollout.log_prob(a_rollout[idx]).sum(-1)
                
                #print(new_probs.size(), old_probs[idx].size())

                log_ratio = (new_probs - old_probs[idx].sum(-1).detach())
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                entropy = dist_rollout.entropy().mean()
                adv = adv_rollout[idx].detach()

                clipped_ratio = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
                actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

                loss = actor_loss + (critic_loss * self.critic_coeff) - (entropy * self.ent_coeff)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

            ### stopping early ###
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        return loss

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        return True

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
        return True

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    #env_name = 'CartPole-v0'
    env = RecordEpisodeStatistics(gym.make(env_name))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    a2c_agent = PPO(
        A2C_Model(obs_dim, act_dim),
        n_steps=256,
        batch_size=16,
        k_epochs=4)

    buffer = Rollout_Memory()

    avg_r = deque(maxlen=20)
    count = 0

    s_t = env.reset()

    for i in range(50000):
        a_t, a_log = a2c_agent.select_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        buffer.push(s_t, a_t, r_t, a_log, done)
        s_t = s_tp1

        if done or buffer.qty == a2c_agent.n_steps:
            r_trajectory = 0
            if not done:
                r_trajectory = a2c_agent.network.critic(torch.from_numpy(s_tp1).float())  

            rollout = buffer.pop_all()
            a2c_agent.update(rollout, r_trajectory)     

            if done:
                count += 1
                s_t = env.reset()
                avg_r.append(int(info["episode"]["r"]))
                #break

        if i % a2c_agent.verbose == 0 and i > 0:
            #avg_r = sum(episodic_rewards)/len(episodic_rewards)
            print(f'Episode: {count} | Average reward: {sum(avg_r)/len(avg_r)} | Timesteps: {i} | [{len(avg_r)}]')
                
if __name__ == "__main__":
   main()

    

