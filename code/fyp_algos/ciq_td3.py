#currently reaches 200+ score on LunarLander in 322 episodes/130,000 timesteps
#bit unstable but slowly gets there
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import gym
import copy
from gym.wrappers import FrameStack
from collections import deque
import random

from code.utils.models import td3_Actor
from code.fyp_algos.ciq_sac import encoder_Critic
from code.utils.attacker import Attacker
from code.fyp_algos.ciq_pt import Transition

class CIQ_TD3():
    def __init__(self, 
        environment, 
        actor,
        critic,
        lr=1e-4,
        buffer_size=1000000, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        train_after=0,
        policy_delay=2,
        target_policy_noise=0.2, 
        target_noise_clip=0.5,
        EPS_END=0.05):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
  
        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.EPS_END = EPS_END
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_t = torch.from_numpy(np.array(batch.t)).type(torch.float32)

        #Critic update
        with torch.no_grad():
            a_p = self.actor_target(s_p)
            a_p = torch.from_numpy(self.noisy_action(a_p))
            target_q = self.critic_target(s_p, a_p, i_t)[0]
            y = r + self.gamma * target_q * (1 - d)

        idx = torch.zeros(len(i_t)).unsqueeze(-1)
        i_ghost = torch.zeros(i_t.size()).scatter(-1, idx.long(), 1)
        q = self.critic(s, a, i_ghost)[0]

        i_p = self.critic(s_p, a_p, i_t)[1]

        i_loss = F.binary_cross_entropy_with_logits(i_p, i_t)
        critic_loss = F.mse_loss(q, y) + i_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            policy_loss = -self.critic.forward(s, self.actor(s), i_ghost)[0].mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss    

    def noisy_action(self, a_t):
        mean=torch.zeros_like(a_t)
        noise = torch.normal(mean=mean, std=0.1).clamp(-self.target_noise_clip, self.target_noise_clip)
        return (a_t + noise).clamp(self.act_low,self.act_high).numpy()

    def select_action(self, s):
        self.EPS = max(self.EPS_END, self.EPS * self.EPS_DECAY)
        if torch.rand(1) > self.EPS:
            a = self.actor(torch.tensor(s).float()).detach()
        else:
            a = torch.from_numpy(self.environment.action_space.sample()).float()

        return a

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'  
    """
    -DDPG with regular critic learns continuous cartpole in ~20,000 timesteps with train_after=10,000
    -adding encoder to the network doesnt completely hinder learning but slows it down a 
        bit and makes it unstable, maybe tweak the size of the encoder
    -a lot to talk about in terms of experimenting with how you transferred ciq to ddpg
    """
    env = gym.make(env_name)
    env = Attacker(env)
    stacks = 2
    env = FrameStack(env, stacks)

    episodic_rewards = deque(maxlen=10)
    episodes = 0

    ddpg_agent = CIQ_TD3(environment=env,    #taken from sb3 zoo
        actor=td3_Actor(8, 1, env.action_space.high[0]),
        critic=encoder_Critic(stacks, 4, 4, 1),
        buffer_size=200000,
        tau=0.01,
        gamma=0.98, 
        train_after=1000)

    replay_buffer = deque(maxlen=ddpg_agent.buffer_size)
    r_sum = 0
    s_t = env.reset()
    s_t = np.concatenate([s_t[0], s_t[1]])

    for i in range(300000):
        a_t = ddpg_agent.actor(torch.from_numpy(s_t).float()).detach()
        a_t = ddpg_agent.noisy_action(a_t)
        
        s_tp1, r_t, done, i_t = env.step(a_t)
        r_sum += r_t
        s_tp1 = np.concatenate([s_tp1[0], s_tp1[1]])
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done, i_t])
        s_t = s_tp1

        if len(replay_buffer) >= 100 and i >= ddpg_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=100)))
            loss = ddpg_agent.update(batch, i)

            if i % 500 == 0:     #for formatting, I want to round it better than just making it an int!
                avg_r = sum(episodic_rewards)/10
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_t = env.reset()
            s_t = np.concatenate([s_t[0], s_t[1]])
        

    #Render Trained agent
    s_t = env.reset()
    while True:
        env.render()
        a_t = np.array(ddpg_agent.actor(torch.from_numpy(s_t)).detach())
        s_tp1, r_t, done, _ = env.step(a_t)
        if done:
            s_tp1 = env.reset()

        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()