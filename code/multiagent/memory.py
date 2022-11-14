import torch
import numpy as np

class ExperienceReplay():
    def __init__(
        self,
        capacity
    ) -> None:
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        pass

    def __len__(
        self
    ) -> int:
        return len(self.dones)

    def sample(
        self,
        idxs
    ) -> list:

        states = torch.from_numpy(np.array(self.states)[idxs]).float()
        actions = torch.from_numpy(np.array(self.actions)[idxs]).float()
        rewards = torch.from_numpy(np.array(self.rewards)[idxs]).float()
        next_states = torch.from_numpy(np.array(self.next_states)[idxs]).float()
        dones = torch.from_numpy(np.array(self.dones)[idxs]).float()

        return states, actions, rewards, next_states, dones

    def store(
        self,
        s_t,
        a_t,
        r_t,
        s_tp1,
        done
    ) -> None:

        if len(self.dones) >= self.capacity:
            self.states.pop()
            self.actions.pop()
            self.rewards.pop()
            self.next_states.pop()
            self.dones.pop()

        self.states.append(s_t)
        self.actions.append(a_t)
        self.rewards.append(r_t)
        self.next_states.append(s_tp1)
        self.dones.append(done)
