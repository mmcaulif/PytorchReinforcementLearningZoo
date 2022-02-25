from collections import deque
from distutils.log import info
import random
from typing import NamedTuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym import Wrapper
from dqn_pt import DQN
from utils.models import Q_val

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int
    t: int

replay_buffer = deque(maxlen=100000)

#network
class CIQ_Net(nn.Module):
	def __init__(self, state_dim):
		super(CIQ_Net, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(state_dim, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 32),
                                     nn.ReLU())

	def ciq_forward(self, state):
		Z = self.encoder(state)

		return Z

#agent, subclass of DQN
class CIQ(DQN):    #test that this functions correctly as a vanilla dqn
    def __init__(self, tau=1e-3, step=4, num_treatment=2):
        self.tau = tau

        self.logits_t = nn.Sequential(nn.Linear(32, 32 // 2),
                                      nn.ReLU(),
                                      nn.Linear(32 // 2, num_treatment))
        
        self.fc = nn.Sequential(nn.Linear((32 + num_treatment) * step , (32 + num_treatment) * step // 2),
                                nn.ReLU(),
                                nn.Linear((32 + num_treatment) * step // 2, 2))

    def soft_update_target(self):
        for self.q_target, self.q_func in zip(self.q_target.parameters(), self.q_func.parameters()):
            self.q_target.data.copy_(self.tau*self.q_func.data + (1.0-self.tau)*self.q_target.data)

    """def ciq_update(self, batch):
        s = torch.from_numpy(np.array(batch.s))
        a = torch.IntTensor(batch.a).unsqueeze(1)
        r = torch.FloatTensor(batch.r).unsqueeze(1)
        s_p = torch.from_numpy(np.array(batch.s_p))
        d = torch.IntTensor(batch.d).unsqueeze(1)
        i_training = torch.IntTensor(batch.t)
        loss = 0
        print(i_training)"""

class CIQ(nn.Module):
    def __init__(self, step=4, num_treatment=2):
        super(CIQ, self).__init__()
        self.step = step, 
        self.num_treatment = num_treatment,
        
        self.encoder = nn.Sequential(nn.Linear(4, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 32),
                                     nn.ReLU())

        self.logits_t = nn.Sequential(nn.Linear(32, 32 // 2),
                                      nn.ReLU(),
                                      nn.Linear(32 // 2, num_treatment))
        
        self.fc = nn.Sequential(nn.Linear((32 + num_treatment) * step , (32 + num_treatment) * step // 2),
                                nn.ReLU(),
                                nn.Linear((32 + num_treatment) * step // 2, 2))

    def forward(self, data):
        out = {}
        #print(data)
        data = [torch.from_numpy(d) for d in data]

        data = [self.encoder(s) for s in data]

        z = data
        t = [self.logits_t(_z) for _z in z]

        z = torch.cat(z, dim=-1)
        print(z)
        z = F.pad(z, pad=(32 * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        print(z)
        t_stack = torch.stack(t, dim=1)

        if self.training:
            _t = torch.stack(data['t'][-self.step:], dim=1)
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, _t.long(), 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
        
        else:
            """
            bit when not training removed for readability
            """

        onehot_t = F.pad(onehot_t, pad=(self.num_treatment * self.step - onehot_t.shape[-1], 0))
        y = self.fc(torch.cat([z, onehot_t], dim=-1))
        
        out['t'] = t[-1]
        out['y'] = y
        out['z'] = z
        return out

class GaussianNoise(Wrapper):
    def __init__(self, env, p=0.5, var=1):
        super().__init__(env)
        self.env = env
        self.p = p
        self.var = var
        self.obs_low = env.observation_space.low[0]
        self.obs_high = env.observation_space.high[0]

    def step(self, action):  
        next_state, reward, done_bool, _ = super().step(action)

        noise = torch.normal(mean=torch.zeros_like(torch.from_numpy(next_state)), std=self.var).numpy()
        
        I = 0
        if torch.rand(1) < self.p: I = 1
        
        gaussian_state = np.clip((next_state + noise), self.obs_low, self.obs_high)
        next_state = I * gaussian_state + (1 - I) * next_state

        return next_state, reward, done_bool, I


env = gym.make('CartPole-v1')
env = GaussianNoise(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
#ciq_agent = CIQ(env, 0.99, 50000, 1000, 64, Q_val(obs_dim, act_dim), 500)


"""for i in range(20):
    a_t = ciq_agent.select_action(s_t)
    
    s_tp1, r_t, d, i_t = env.step(a_t)

    #print(s_tp1, i_t)

    replay_buffer.append([s_t, a_t, r_t, s_tp1, d, i_t])
    
    s_t = s_tp1

    if d:
        s_t = env.reset()"""

def main():
    obs = deque(maxlen=4)
    ciq_agent = CIQ()
    s_t = env.reset()

    for i in range(4):
        obs.append(s_t)
        a_t = env.action_space.sample()
        s_tp1, r_t, d, _ = env.step(a_t)

        if len(obs) >= 4:
            print("out")
            with torch.no_grad():
                out = ciq_agent(obs)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
