from collections import deque
import random
from typing import NamedTuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gym import Wrapper
#from dqn_pt import DQN
#from utils.models import Q_val

class Transition(NamedTuple):
    s: list
    a: float
    r: float
    s_p: list
    d: int
    i: int

replay_buffer = deque(maxlen=100000)

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

    def forward(self, data, t_labels):
        #Data comes in a list of length 'step' which is stacked observations
        #print(f"Data: {data}")
        #data = [torch.from_numpy(d) for d in data]
        data = torch.as_tensor(data)    #imo looks better but is slow as data is a deque of np arrays

        data = self.encoder(data).flatten()
        #comes out as a flattend tensor of length 128 (step * 32)
        
        z = data
        t = z.view(self.step[0], 32)
        t = self.logits_t(t)    #outputs as a step * num treatments tensor
        
        print(len(z), z.size())
        z = F.pad(z, pad=(32 * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        #   ^ to be fixed, https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        
        t_stack = torch.stack(t, dim=1)

        if self.training:   #create one hot vectors denoting the treatment labels
            _t = torch.stack(t_labels[-self.step:], dim=1)
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, _t.long(), 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
        
        else:
            """
            bit when not training removed for readability
            """

        onehot_t = F.pad(onehot_t, pad=(self.num_treatment * self.step - onehot_t.shape[-1], 0))
        y = self.fc(torch.cat([z, onehot_t], dim=-1))
        
        return t[-1], y, z

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

def ciq_loss(q, y, i_p, i_t):
    t_onehot = torch.zeros(i_p.shape).type(i_p.type())
    t_onehot = t_onehot.scatter(1, i_t.long(), 1)
    return F.mse_loss(q, y) + nn.BCEWithLogitsLoss()(i_p, t_onehot)

def ciq_update(batch, network, optimizer):
    s = torch.from_numpy(np.array(batch.s)).type(torch.float32)
    a = torch.from_numpy(np.array(batch.a)).unsqueeze(1)
    r = torch.FloatTensor(batch.r).unsqueeze(1)
    s_p = torch.from_numpy(np.array(batch.s_p)).type(torch.float32)
    d = torch.IntTensor(batch.d).unsqueeze(1)
    i = torch.Tensor(batch.i)

    i_model, q, _ = network(s, i)
    q = torch.gather(q, 1, a.long())

    i_p = torch.zeros_like(i)
    _, q_p, _ = network(s_p, i_p)
    y = r + 0.99 * q_p.max(1)[0].view(4, 1) * (1 - d)

    loss = ciq_loss(q, y, i_model, i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

env = gym.make('CartPole-v1')
env = GaussianNoise(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n



def main():
    #obs = deque(maxlen=4)
    ciq_net = CIQ(step=1)
    criterion = optim.Adam(ciq_net.parameters(), lr=1e-3)
    s_t = env.reset()

    for t in range(4):
        #obs.append(s_t)
        a_t = env.action_space.sample()
        s_tp1, r_t, d, i_t = env.step(a_t)
        replay_buffer.append([s_t, a_t, r_t, s_tp1, d, i_t])


        ciq_net(torch.from_numpy(s_t).type(torch.float32), torch.Tensor(i_t))

        """if len(replay_buffer) >= 4:
            batch = Transition(*zip(*random.sample(replay_buffer, k=4)))
            ciq_update(batch, ciq_net, criterion)"""

        if d:
            s_tp1 = env.reset()

        s_t = s_tp1

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
