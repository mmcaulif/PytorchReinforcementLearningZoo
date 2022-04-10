import torch
import numpy as np
from gym import Wrapper

class Attacker(Wrapper):
    def __init__(self, env, p=0.1, var=0.5):
        super().__init__(env)
        self.env = env
        self.p = p
        self.var = var
        self.obs_low = env.observation_space.low[0]
        self.obs_high = env.observation_space.high[0]
        self.obs_dims = env.observation_space.n
        self.prev_state = np.zeros(self.obs_dims)

        return 3    #number of pssible treatments

    def step(self, action):  
        I = 0
        next_state, reward, done_bool, _ = super().step(action)
        t_labels = np.zeros(3)
        noise = torch.normal(mean=torch.zeros_like(torch.from_numpy(next_state)), std=self.var).numpy()
        rnd = torch.rand(1)

        if 0 < rnd < self.p/2: #gaussian noise
            I = 1
            gaussian_state = np.clip((next_state + noise), self.obs_low, self.obs_high)
            next_state = gaussian_state

        elif self.p/2 < rnd < self.p: #blackout
            I = 2
            next_state = np.zeros(self.obs_dims)

        else: 
            t_labels[I] = 1

        return next_state, reward, done_bool, t_labels