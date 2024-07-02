import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def func_hidden_init(data_layer):
    data_fan_in = data_layer.weight.data.size()[0]
    data_lim = 1. / np.sqrt(data_fan_in)
    return (-data_lim, data_lim)

class ModuleActor(nn.Module):
    def __init__(self, data_state_size, data_action_size, data_seed, data_fc1_units=128, data_fc2_units=128):
        super(ModuleActor, self).__init__()
        self.data_seed = torch.manual_seed(data_seed)
        self.data_fc1 = nn.Linear(data_state_size, data_fc1_units)
        self.data_fc2 = nn.Linear(data_fc1_units, data_fc2_units)
        self.data_fc3 = nn.Linear(data_fc2_units, data_action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.data_fc1.weight.data.uniform_(*func_hidden_init(self.data_fc1))
        self.data_fc2.weight.data.uniform_(*func_hidden_init(self.data_fc2))
        self.data_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, data_state):
        data_x = F.relu(self.data_fc1(data_state))
        data_x = F.relu(self.data_fc2(data_x))
        return F.tanh(self.data_fc3(data_x))


class ModuleCritic(nn.Module):
    def __init__(self, data_state_size, data_action_size, data_seed, data_fcs1_units=128, data_fc2_units=128):
        super(ModuleCritic, self).__init__()
        self.data_seed = torch.manual_seed(data_seed)
        self.data_fcs1 = nn.Linear(data_state_size, data_fcs1_units)
        self.data_fc2 = nn.Linear(data_fcs1_units + data_action_size, data_fc2_units)
        self.data_fc3 = nn.Linear(data_fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.data_fcs1.weight.data.uniform_(*func_hidden_init(self.data_fcs1))
        self.data_fc2.weight.data.uniform_(*func_hidden_init(self.data_fc2))
        self.data_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, data_state, data_action):
        data_xs = F.relu(self.data_fcs1(data_state))
        data_x = torch.cat((data_xs, data_action), dim=1)
        data_x = F.relu(self.data_fc2(data_x))
        return self.data_fc3(data_x)

