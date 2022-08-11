from typing import Tuple

import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, state_shape: np.shape, device: str, alpha: float = 0.7, eps: float = 0.1):
        self.max_size = max_size
        self.device = device
        self.alpha = alpha
        self.eps = eps

        if isinstance(state_shape, int):
            state_shape = (state_shape, )

        self.states = torch.zeros((max_size, *state_shape), device=device, dtype=torch.float)
        self.actions = torch.zeros((max_size, 1), device=device, dtype=torch.long)
        self.rewards = torch.zeros((max_size, 1), device=device, dtype=torch.float)
        self.next_states = torch.zeros((max_size, *state_shape), device=device, dtype=torch.float)
        self.dones = torch.zeros((max_size, 1), device=device, dtype=torch.long)
        self.priorities = torch.zeros(max_size, device=device, dtype=torch.float)

        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.from_numpy(state)
        self.actions[self.position][0] = action
        self.rewards[self.position][0] = reward
        self.next_states[self.position] = torch.from_numpy(next_state)
        self.dones[self.position][0] = done
        self.priorities[self.position] = max(torch.max(self.priorities), 1.0)
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        scaled_priorities = self.priorities ** self.alpha
        probabilities = scaled_priorities / torch.sum(scaled_priorities)
        indices = torch.multinomial(probabilities[:self.size], batch_size).to(self.device)

        importance = (1 / self.size * 1 / probabilities[indices]).unsqueeze(1)
        importance /= importance.max()

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return (states, actions, rewards, next_states, dones), indices, importance

    def update(self, indices: np.array, errors: torch.Tensor):
        self.priorities[indices] = errors.squeeze(1) + self.eps
