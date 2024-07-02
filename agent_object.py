import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_module import ModuleActor, ModuleCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0
GAMMA = 0.99
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ObjectAgent():    
    def __init__(self, data_state_size, data_action_size, data_n_agents, data_random_seed):
        self.data_state_size = data_state_size
        self.data_action_size = data_action_size
        self.data_seed = random.seed(data_random_seed)
        self.data_actor_local = ModuleActor(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_actor_target = ModuleActor(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_actor_optimizer = optim.Adam(self.data_actor_local.parameters(), lr=LR_ACTOR)
        self.data_critic_local = ModuleCritic(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_critic_target = ModuleCritic(data_state_size, data_action_size, data_random_seed).to(device)
        self.data_critic_optimizer = optim.Adam(self.data_critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.data_noise = ObjectOUNoise((data_n_agents, data_action_size), data_random_seed)
        self.data_memory = ObjectReplayBuffer(data_action_size, BUFFER_SIZE, BATCH_SIZE, data_random_seed)

    def learn(self, data_experiences, data_gamma):
        data_states, data_actions, data_rewards, data_next_states, data_dones = data_experiences
        data_actions_next = self.data_actor_target(data_next_states)
        data_Q_targets_next = self.data_critic_target(data_next_states, data_actions_next)
        data_Q_targets = data_rewards + (data_gamma * data_Q_targets_next * (1 - data_dones))
        data_Q_expected = self.data_critic_local(data_states, data_actions)
        data_critic_loss = F.mse_loss(data_Q_expected, data_Q_targets)
        self.data_critic_optimizer.zero_grad()
        data_critic_loss.backward()
        self.data_critic_optimizer.step()
        data_actions_pred = self.data_actor_local(data_states)
        data_actor_loss = -self.data_critic_local(data_states, data_actions_pred).mean()
        self.data_actor_optimizer.zero_grad()
        data_actor_loss.backward()
        self.data_actor_optimizer.step()
        self.soft_update(self.data_critic_local, self.data_critic_target, TAU)
        self.soft_update(self.data_actor_local, self.data_actor_target, TAU)                     

    def soft_update(self, data_local_model, data_target_model, data_tau):
        for target_param, local_param in zip(data_target_model.parameters(), data_local_model.parameters()):
            target_param.data.copy_(data_tau*local_param.data + (1.0-data_tau)*target_param.data)
    
    def step(self, data_states, data_actions, data_rewards, data_next_states, data_dones):
        for i_state, i_action, i_reward, i_next_state, i_done in zip(data_states, data_actions, data_rewards, data_next_states, data_dones):
            self.data_memory.add(i_state, i_action, i_reward, i_next_state, i_done)
        if len(self.data_memory) > BATCH_SIZE:
            data_experiences = self.data_memory.sample()
            self.learn(data_experiences, GAMMA)

    def reset(self):
        self.data_noise.reset()

    def act(self, data_state, data_add_noise=True):
        data_state = torch.from_numpy(data_state).float().to(device)
        self.data_actor_local.eval()
        with torch.no_grad():
            data_action = self.data_actor_local(data_state).cpu().data.numpy()
        self.data_actor_local.train()
        if data_add_noise:
            data_action += self.data_noise.sample()
        return np.clip(data_action, -1, 1)

class ObjectReplayBuffer:
    def __init__(self, data_action_size, data_buffer_size, data_batch_size, data_seed):
        self.data_action_size = data_action_size
        self.data_memory = deque(maxlen=data_buffer_size)
        self.data_batch_size = data_batch_size
        self.data_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.data_seed = random.seed(data_seed)
    
    def sample(self):
        data_experiences = random.sample(self.data_memory, k=self.data_batch_size)

        data_states = torch.from_numpy(np.vstack([e.data_state for e in data_experiences if e is not None])).float().to(device)
        data_actions = torch.from_numpy(np.vstack([e.data_action for e in data_experiences if e is not None])).float().to(device)
        data_rewards = torch.from_numpy(np.vstack([e.data_reward for e in data_experiences if e is not None])).float().to(device)
        data_next_states = torch.from_numpy(np.vstack([e.data_next_state for e in data_experiences if e is not None])).float().to(device)
        data_dones = torch.from_numpy(np.vstack([e.data_done for e in data_experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (data_states, data_actions, data_rewards, data_next_states, data_dones)

    def __len__(self):
        return len(self.data_memory)
    
    def add(self, data_state, data_action, data_reward, data_next_state, data_done):
        data_e = self.data_experience(data_state, data_action, data_reward, data_next_state, data_done)
        self.data_memory.append(data_e)

class ObjectOUNoise:
    def __init__(self, data_size, data_seed, data_mu=0., data_theta=0.15, data_sigma=0.2):
        self.data_size = data_size
        self.data_mu = data_mu * np.ones(data_size)
        self.data_theta = data_theta
        self.data_sigma = data_sigma
        self.data_seed = random.seed(data_seed)
        self.reset()

    def sample(self):
        data_x = self.data_state
        data_dx = self.data_theta * (self.data_mu - data_x) + self.data_sigma * np.random.standard_normal(self.data_size)
        self.data_state = data_x + data_dx
        return self.data_state

    def reset(self):
        self.data_state = copy.copy(self.data_mu)