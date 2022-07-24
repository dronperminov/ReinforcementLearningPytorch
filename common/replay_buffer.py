from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        replay = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(replay)
        else:
            self.buffer[self.position] = replay
            self.position = (self.position + 1) % self.max_size

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experience = np.random.choice(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for v in experience:
            states.append(v['state'])
            actions.append(v['action'])
            rewards.append(v['reward'])
            next_states.append(v['next_state'])
            dones.append(v['done'])

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones)).float().to(device)

        return states, actions, rewards, next_states, dones
