"""
Causal inference Q-network for my final year project of exploring a variety of Causal-inference
based augmentations to deep reinforcement learning algorithmns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
#from gym import

"""
Taken from the CIQ paper's github
"""

class AbstractDQN(nn.Module):
    def __init__(self, state_size=4, action_size=2, fc1_units=32, fc2_units=32):
        super(AbstractDQN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(state_size, fc1_units),
                                     nn.ReLU(),
                                     nn.Linear(fc1_units, fc2_units),
                                     nn.ReLU())

    def _forward(self, data):
        raise NotImplementedError

    def forward(self, data):
        data['z'] = [self.encoder(s) for s in data['state']]
        out = self._forward(data)
        return out


class CEQNetwork_1(AbstractDQN):
    """ this model use (1) step * frame and (2) treatment; then concat (1) and (2) together. Using fc to predict Q
    """ 
    def __init__(self, state_size=4, action_size=2, fc1_units=32, fc2_units=32, step=4, num_treatment=2):
        super(CEQNetwork_1, self).__init__(state_size, action_size, fc1_units, fc2_units)
        self.name = 'CEQNetwork_1'
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.num_treatment = num_treatment
        self.step = step
        self.logits_t = nn.Sequential(nn.Linear(fc2_units, fc2_units // 2),
                                      nn.ReLU(),
                                      nn.Linear(fc2_units // 2, num_treatment))

        self.fc = nn.Sequential(nn.Linear((fc2_units + num_treatment) * step , (fc2_units + num_treatment) * step // 2),
                                nn.ReLU(),
                                nn.Linear((fc2_units + num_treatment) * step // 2, action_size))

    def _forward(self, data):
        out = {}
        z = data['z'][-self.step:]
        t = [self.logits_t(_z) for _z in z]

        z = torch.cat(z, dim=-1)
        z = F.pad(z, pad=(self.fc2_units * self.step - z.shape[-1], 0)) # pad zeros to the left to fit in fc layer
        t_stack = torch.stack(t, dim=1)
        
        if self.training:
            _t = torch.stack(data['t'][-self.step:], dim=1)
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, _t.long(), 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)
        else:
            onehot_t = torch.zeros(t_stack.shape).type(t_stack.type())
            onehot_t = onehot_t.scatter(2, t_stack.topk(1, 2, True, True)[1], 1)
            onehot_t = onehot_t.view(onehot_t.shape[0], -1)

        onehot_t = F.pad(onehot_t, pad=(self.num_treatment * self.step - onehot_t.shape[-1], 0))
        y = self.fc(torch.cat([z, onehot_t], dim=-1))
        
        out['t'] = t[-1]
        out['y'] = y
        out['z'] = z
        return out
