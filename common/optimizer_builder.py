from typing import Iterator
import torch


class OptimizerBuilder:
    @staticmethod
    def build(name: str, learning_rate: float, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if name == 'sgd':
            return torch.optim.SGD(parameters, lr=learning_rate)

        if name == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate)

        if name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=learning_rate)

        raise ValueError(f"Unknown optimizer \"{name}\"")
