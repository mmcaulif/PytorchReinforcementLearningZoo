import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import numpy as np
import gym
import copy
from gym.wrappers import FrameStack
from collections import deque
import random

import wandb

from code.value_iter.sac_pt import SAC
from code.utils.models import td3_Critic, sac_Actor
from code.utils.attacker import Attacker
from code.fyp_algos.ciq_pt import Transition

class encoder1_Critic(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super(encoder1_Critic, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(obs_dims+act_dims, 400),
                                     nn.ReLU(),
                                     )

        self.fc = nn.Sequential(nn.Linear(400, 300),
                                nn.ReLU(),
                                nn.Linear(300, 1)
                                )
        
    def forward(self, state, action):        
        q = self.fc(self.encoder(torch.cat([state, action], 1)))
        return q, q
    
    def q1_forward(self, state, action):
        q = self.fc(self.encoder(torch.cat([state, action], 1)))
        return q

class encoder2_Critic(nn.Module):
    def __init__(self, num_treatment, obs_dims, act_dims):
        super(encoder2_Critic, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(obs_dims+act_dims, 400),
                                     nn.ReLU(),
                                     )

        self.logits_t = nn.Sequential(nn.Linear(400, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, num_treatment)
                                      )

        self.fc = nn.Sequential(nn.Linear(400+num_treatment, 300),
                                nn.ReLU(),
                                nn.Linear(300, 1)
                                )
        
    def forward(self, state, action, t_labels):
        z = self.encoder(torch.cat([state, action], 1))

        t_values = self.logits_t(z)
        idx = torch.argmax(t_values, dim=-1).long().unsqueeze(-1)
        t_p = torch.zeros_like(t_values).scatter(-1, idx, 1)
        
        if self.training:
            q = self.fc(torch.cat([z, t_labels], dim=-1))
        else:
            q = self.fc(torch.cat([z, t_p], dim=-1))

        return q, t_labels#, t_p    #altered for debugging purposes

class CIQ_SAC():
    def __init__(self, 
        environment, 
        actor,
        critic,
        lr=3e-4,
        buffer_size=1000000, 
        batch_size=256, 
        alpha=0.2,
        tau=0.005, 
        gamma=0.99, 
        train_after=100,
        policy_delay=1,
        verbose=500):

        self.environment = environment
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.train_after = train_after
        self.policy_delay = policy_delay
        self.verbose = verbose

        self.act_high = environment.action_space.high[0]
        self.act_low = environment.action_space.low[0]

        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # action rescaling
        #print(self.act_high, self.act_low)
        self.action_scale = int((self.act_high - (self.act_low)) / 2.0)
        self.action_bias = int((self.act_high + (self.act_low)) / 2.0)

    def update(self, batch, i):
        s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
        a = torch.from_numpy(np.array(batch.a))
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_t = torch.from_numpy(np.array(batch.t)).type(torch.float32)

        #Critic update
        with torch.no_grad():
            a_p, log_pi, _ = self.select_action(s_p)
            target_q = self.critic_target(s_p, a_p, i_t)[0] - (log_pi * self.alpha)
            y = r + self.gamma * target_q * (1 - d)
        
        idx = torch.zeros(len(i_t)).unsqueeze(-1)
        i_ghost = torch.zeros(i_t.size()).scatter(-1, idx.long(), 1)
        q = self.critic(s, a, i_ghost)[0]

        i_p = self.critic(s_p, a_p, i_t)[1]

        i_loss = F.binary_cross_entropy_with_logits(i_p, i_t)
        critic_loss = F.mse_loss(q, y) + i_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        if i % self.policy_delay == 0:
            a_p, log_pi, _ = self.select_action(s)
            policy_loss = -(self.critic(s, a_p, i_ghost)[0] - (log_pi * self.alpha)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss

    def select_action(self, s):
        mean, log_std = self.actor(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def act(self, s):
        a = self.select_action(torch.from_numpy(s).float())[0].detach().numpy()
        return a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probs", help="Probability of interference", type=float)
    parser.add_argument("--vanilla", help="TD3 or TD3_ciq", action="store_true")
    args = parser.parse_args()

    wandb.init(project="fyp-td3-ciq", entity="manusft")
    wandb.config = {
        "algo": 'SAC',
        "vanilla": args.vanilla,
        "P": args.probs
        }

    env_name = 'gym_cartpole_continuous:CartPoleContinuous-v0'
    env = gym.make(env_name)
    env = Attacker(env, p=args.probs)
    stacks = 1
    #env = FrameStack(env, stacks)
    
    obs_dims = env.observation_space.shape[0] * stacks
    act_dims = env.action_space.shape[0]
    
    episodic_rewards = deque(maxlen=10)
    episodes = 0
    LR = 2e-3

    """
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    """

    if args.vanilla:
        print(f"Using vanilla sac with P: {args.probs}")
        sac_agent = SAC(environment=env,    #taken from sb3 zoo
            actor=sac_Actor(obs_dims, act_dims),
            #critic=encoder1_Critic(obs_dims, act_dims),
            critic=td3_Critic(obs_dims, act_dims),
            lr=LR,
            buffer_size=200000,
            train_after=10000,
            alpha=0.15
            )
    
    else:
        print(f"Using ciq_sac with P: {args.probs}")
        sac_agent = CIQ_SAC(environment=env,    #taken from sb3 zoo
            actor=sac_Actor(obs_dims, act_dims),
            critic=encoder2_Critic(4, obs_dims, act_dims),
            lr=LR,
            buffer_size=200000,
            train_after=10000,
            alpha=0.15
            )

    replay_buffer = deque(maxlen=sac_agent.buffer_size)

    r_sum = 0
    s_t = env.reset()
    #s_t = np.concatenate([s_t[0], s_t[1]])

    for i in range(25000+1):
        if i >= sac_agent.train_after:
            a_t = sac_agent.act(s_t)
        else:
            a_t = env.action_space.sample()
        
        s_tp1, r_t, done, i_t = env.step(a_t)
        r_sum += r_t
        #s_tp1 = np.concatenate([s_tp1[0], s_tp1[1]])
        replay_buffer.append([s_t, a_t, r_t, s_tp1, done, i_t])

        if len(replay_buffer) >= sac_agent.batch_size and i >= sac_agent.train_after:
            batch = Transition(*zip(*random.sample(replay_buffer, k=sac_agent.batch_size)))
            loss = sac_agent.update(batch, i)

            if i % sac_agent.verbose == 0: 
                avg_r = sum(episodic_rewards)/len(episodic_rewards)
                wandb.log({f"Avg Reward with P={args.probs} (ciq agumented vs vanilla with encoder critic)": avg_r})                
                #wandb.log({f"Avg Reward with P={args.probs}, lr={LR} (using encoder critic network and oracle loss)": avg_r})
                print(f"Episodes: {episodes} | Timestep: {i} | Avg. Reward: {avg_r}, [{len(episodic_rewards)}]")

        if done:
            episodes += 1
            episodic_rewards.append(r_sum)
            r_sum = 0
            s_tp1 = env.reset()
            #s_tp1 = np.concatenate([s_tp1[0], s_tp1[1]])

        s_t = s_tp1

    #Render Trained agent
    """s_t = env.reset()
    while True:
        env.render()
        a_t = sac_agent.act(s_t)
        s_tp1, r_t, done, _ = env.step(a_t)
        if done:
            s_tp1 = env.reset()

        s_t = s_tp1"""

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()