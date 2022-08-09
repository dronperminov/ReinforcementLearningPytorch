from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size: int, state_shape: np.shape, device: str):
        self.max_size = max_size
        self.device = device

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        self.states = torch.zeros((max_size, *state_shape), device=device, dtype=torch.float)
        self.actions = torch.zeros((max_size, 1), device=device, dtype=torch.long)
        self.rewards = torch.zeros((max_size, 1), device=device, dtype=torch.float)
        self.next_states = torch.zeros((max_size, *state_shape), device=device, dtype=torch.float)
        self.dones = torch.zeros((max_size, 1), device=device, dtype=torch.long)

        self.position = 0
        self.size = 0

    def clear(self):
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.from_numpy(state)
        self.actions[self.position][0] = action
        self.rewards[self.position][0] = reward
        self.next_states[self.position] = torch.from_numpy(next_state)
        self.dones[self.position][0] = done
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones
