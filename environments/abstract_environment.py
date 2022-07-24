from typing import Tuple
import abc
import numpy as np


class AbstractEnvironment:
    def __init__(self, aspect_ratio: float):
        self.aspect_ratio = aspect_ratio
        self.action_space_shape = None
        self.observation_space_shape = None

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    @abc.abstractmethod
    def draw(self, width: int, height: int) -> np.ndarray:
        pass
