import torch
import numpy as np
import gym
from gym import Wrapper

class Attacker(Wrapper):
    def __init__(self, env, p=0.1, var=0.5):
        super().__init__(env)
        self.env = env
        self.p = p
        self.var = var
        self.obs_low = env.observation_space.low[0]
        self.obs_high = env.observation_space.high[0]
        self.obs_dims = env.observation_space.shape[0]
        self.prev_state = np.zeros(self.obs_dims)
        self.num_treatments = 4

    def step(self, action):  
        next_state, reward, done_bool, _ = super().step(action)
        self.next_state = next_state

        t_labels = np.zeros(self.num_treatments)
        
        rnd = torch.rand(1)

        if 0 < rnd < self.p/3: #gaussian noise
            I = 1
            noise = torch.normal(mean=torch.zeros_like(torch.from_numpy(next_state)), std=self.var).numpy()
            gaussian_state = np.clip((next_state + noise), self.obs_low, self.obs_high)
            next_state = gaussian_state

        elif self.p/3 < rnd < 2*(self.p/3): #blackout
            I = 2
            next_state = np.zeros(self.obs_dims)
            self.next_state = next_state

        elif 2*(self.p/3) < rnd < self.p:   #lag
            I = 3
            next_state = self.prev_state

        else:
            I = 0

        self.prev_state = self.next_state

        t_labels[I] = 1

        return next_state, reward, done_bool, t_labels

def main():
    env = gym.make('CartPole-v0')
    env = Attacker(env, p=0.5)

    s_t = env.reset()

    for i in range(50):
        a_t = env.action_space.sample()
        s_tp1, _, d, i_t = env.step(a_t)
        print(f'{i}: Obs: {s_tp1} \nTreatment labels: {i_t}\n')
        s_t = s_tp1
        if d:
            s_t = env.reset()

if __name__ == "__main__":
    main()