from environments.abstract_environment import AbstractEnvironment
from algorithms.abstract_algorithm import AbstractAlgorithm
from numpy.random import randint


class RandomAlgorithm(AbstractAlgorithm):
    def __init__(self, environment: AbstractEnvironment, config: dict):
        super().__init__(environment, config)

    def run(self, draw: bool = True):
        self.environment.reset()

        episode_reward = 0
        done = False
        info = {}

        while not done:
            action = randint(self.environment.action_space_shape)
            next_state, reward, done, info = self.environment.step(action)

            if draw:
                self.draw()

            episode_reward += reward

        self.end_episode(episode_reward, info)
