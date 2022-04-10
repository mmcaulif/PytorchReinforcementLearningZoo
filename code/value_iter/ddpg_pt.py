#currently reaches 200+ score on LunarLander in 322 episodes/130,000 timesteps
#bit unstable but slowly gets there
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from gym.wrappers import RecordEpisodeStatistics
from utils.models import td3_Actor, ddpg_Critic

from collections import deque
from typing import NamedTuple
import random
class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

class DDPG():
    def __init__(self, 
        environment, 
        actor,
        critic,
        pi_lr=1e-4, 
        c_lr=1e-3, 
        buffer_size=1000000, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        train_after=0,
        policy_delay=2,
        target_policy_noise=0.2, 
        target_noise_clip=0.5,
        EPS_END=0.05,
        debug_dim=[],
        debug_act_high=[],
        debug_act_low=[]):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        try:
            obs_dim = environment.observation_space.shape[0]            
            act_dim = environment.action_space.shape[0]    
            self.act_high = environment.action_space.high[0]
            self.act_low = environment.action_space.low[0]    
        except:
            obs_dim = debug_dim[0]
            act_dim = debug_dim[1]
            self.act_high = torch.tensor(debug_act_high)
            self.act_low = torch.tensor(debug_act_low)

        self.actor = actor(obs_dim, act_dim, self.act_high)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.critic = critic(obs_dim, act_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.EPS_END = EPS_END
        self.EPS = 0.9
        self.EPS_DECAY = 0.999

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)

        #Critic update
        with torch.no_grad():
            a_p = self.actor_target(s_p)
            a_p = torch.from_numpy(self.noisy_action(a_p))
            target_q = self.critic_target(s_p, a_p)
            y = r + self.gamma * target_q * (1 - d)

        q = self.critic(s, a)

        critic_loss = F.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #delayed Actor update
        if i % self.policy_delay == 0:
            policy_loss = -self.critic.forward(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
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
            #a = self.actor(torch.tensor(s).float()).detach()
            a = self.actor(torch.from_numpy(s).type(torch.float32)).detach()    #might be faster?
        else:
            a = torch.from_numpy(self.environment.action_space.sample()).type(torch.float32)

        return a

class encoder_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(encoder_Critic, self).__init__()

        enc_dim = 64
        num_treatment = 2

        self.critic = nn.Sequential(
            nn.Linear(enc_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, enc_dim),
            #nn.ReLU(),
            #nn.Linear(enc_dim, enc_dim)
        )

        self.oracle = nn.Sequential(
            nn.Linear(enc_dim, enc_dim//2),
            nn.ReLU(),
            nn.Linear(enc_dim//2, num_treatment)
        )

    def forward(self, state, action):
        s_z = self.encoder(state)
        try:
            sa = torch.cat([s_z, action], 1)
        except:	
            sa = torch.cat([s_z, action], -1)

        i_pi = self.oracle(s_z)
        q = self.critic(sa)
        return q#, i_pi

def main():
    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'  
    """
    -DDPG with regular critic learns continuous cartpole in ~20,000 timesteps with train_after=10,000
    -adding encoder to the network doesnt completely hinder learning but slows it down a 
        bit and makes it unstable, maybe tweak the size of the encoder
    -a lot to talk about in terms of experimenting with how you transferred ciq to ddpg
    """
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    c_losses = deque(maxlen=100)
    episodic_rewards = deque(maxlen=10)
    episodes = 0

    ddpg_agent = DDPG(environment=env,    #taken from sb3 zoo
        actor=td3_Actor,
        critic=encoder_Critic, 
        pi_lr=0.000075, #lower LR prevents for the agent from spectacularly forget
        c_lr=0.00075,
        buffer_size=200000, 
        gamma=0.98, 
        train_after=10000,
        target_policy_noise=0.1,
        EPS_END=0.1)

    replay_buffer = deque(maxlen=ddpg_agent.buffer_size)

    s_t = env.reset()

    for i in range(300000):
        a_t = ddpg_agent.actor(torch.from_numpy(s_t).float()).detach()
        a_t = ddpg_agent.noisy_action(a_t)
        
        s_tp1, r_t, done, info = env.step(a_t)
        
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        if len(replay_buffer) >= 100 and i > ddpg_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=100)))
            loss = ddpg_agent.update(batch, i)

            if i % 500 == 0: 
                c_losses.append(loss)
                avg_c_losses = sum(c_losses)/100    #for formatting, I want to round it better than just making it an int!
                avg_r = sum(episodic_rewards)/10
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Critic Loss: {int(avg_c_losses)} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(int(info['episode']['r']))
            s_tp1 = env.reset()

        s_t = s_tp1

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