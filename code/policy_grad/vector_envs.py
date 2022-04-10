import gym
import torch
import numpy as np

#https://tristandeleu.github.io/gym/vector/getting_started.html

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.qty = 0
    
    def push(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.qty += 1
    
    def pop_all(self):
        states = torch.as_tensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.as_tensor(self.rewards).float()
        
        self.states, self.actions, self.rewards = [], [], []
        self.qty = 0
        
        return states, actions, rewards

if __name__ == "__main__":
   envs = gym.vector.make("CartPole-v1", num_envs=4)
   buffer = Memory()

   envs.reset()
   for i in range(1):
        #envs.render()
        a_t = envs.action_space.sample()
        s_t, r_t, d, _ = envs.step(a_t)
        print("Returns: ", s_t, r_t, d)
        
        [buffer.push(s_t[i], r_t[i], d[i]) for i in range(4)]
        print("\nBuffer output: ", buffer.pop_all())
