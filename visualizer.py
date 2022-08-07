import pygame
from pygame.locals import DOUBLEBUF, QUIT, KEYDOWN, KEYUP
import threading

from environments.abstract_environment import AbstractEnvironment
from environments.snake import Snake

from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.random_algorithm import RandomAlgorithm
from algorithms.dqn import DQN
from algorithms.reinforce import Reinforce
from algorithms.actor_critic import ActorCritic
from algorithms.a2c import AdvantageActorCritic
from algorithms.ppo import ProximalPolicyOptimization
from algorithms.gae import GeneralizedAdvantageEstimation


class Visualizer:
    def __init__(self, config: dict):
        if "environment" not in config:
            raise ValueError("Environment is not set")

        if "algorithms" not in config:
            raise ValueError("Algorithms is not set")

        self.__init_view(config.get('view_config', {}), len(config["algorithms"]))
        self.__init_algorithms(config)

    def __set_common_params(self, algorithm_config: dict, common_algorithm_params: dict):
        for key, value in common_algorithm_params.items():
            if key in algorithm_config:
                print(f'Warning: key "{key}" already set for algorithm "{algorithm_config["name"]}"')
                continue

            algorithm_config[key] = value

    def __get_environment(self, environment_config: dict) -> AbstractEnvironment:
        environment_name = environment_config["name"]

        if environment_name == "snake":
            return Snake(**environment_config.get("config", {}))

        raise ValueError(f"Unknown environment \"{environment_name}\"")

    def __get_algorithm(self, algorithm_config: dict, environment: AbstractEnvironment, i: int, j: int):
        algorithm_name = algorithm_config["name"]
        algorithm_config['plot_rewards'] = self.plot_rewards
        algorithm_config['plot_keys'] = self.plot_keys
        algorithm_config['info_keys'] = self.info_keys
        algorithm_config['width'] = self.width
        algorithm_config['height'] = self.height
        algorithm_config['avg_len'] = self.avg_len
        algorithm_config['x0'] = j * self.algorithm_width
        algorithm_config['y0'] = i * self.algorithm_height
        algorithm_config['screen'] = self.screen

        if algorithm_name == "random":
            return RandomAlgorithm(environment, algorithm_config)

        if algorithm_name == "dqn":
            return DQN(environment, algorithm_config)

        if algorithm_name == "reinforce":
            return Reinforce(environment, algorithm_config)

        if algorithm_name == "actor_critic":
            return ActorCritic(environment, algorithm_config)

        if algorithm_name == "a2c":
            return AdvantageActorCritic(environment, algorithm_config)

        if algorithm_name == "ppo":
            return ProximalPolicyOptimization(environment, algorithm_config)

        if algorithm_name == "gae":
            return GeneralizedAdvantageEstimation(environment, algorithm_config)

        raise ValueError(f"Unknown algorithm \"{algorithm_name}\"")

    def __init_view(self, view_config: dict, algorithms_count: int):
        self.plot_rewards = view_config.get('plot_rewards', True)
        self.plot_keys = view_config.get('plot_keys', [])
        self.info_keys = view_config.get('info_keys', [])
        self.avg_len = view_config.get('avg_len', 50)
        self.width = view_config.get('width', 500)
        self.height = view_config.get('height', 500)
        self.columns = view_config.get('columns', 1)
        self.algorithm_width = self.width * (1 + len(self.plot_keys) + self.plot_rewards)
        self.algorithm_height = self.height + 25 + (25 if len(self.info_keys) else 0)
        self.use_gui = view_config.get('use_gui', True)
        self.screen = None

        if not self.use_gui:
            return

        pygame.init()
        pygame.display.set_caption(view_config.get('caption', 'Reinforcement learning sandbox'))
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP])

        self.total_width = self.algorithm_width * self.columns
        self.total_height = self.algorithm_height * ((algorithms_count + self.columns - 1) // self.columns)
        self.screen = pygame.display.set_mode((self.total_width, self.total_height), DOUBLEBUF, 16)
        self.screen.set_alpha(None)

    def __init_algorithms(self, config: dict):
        self.algorithms = []

        for index, algorithm_config in enumerate(config["algorithms"]):
            self.__set_common_params(algorithm_config, config.get('common_algorithms_params', {}))
            i, j = index // self.columns, index % self.columns
            environment = self.__get_environment(config['environment'])
            algorithm = self.__get_algorithm(algorithm_config, environment, i, j)
            self.algorithms.append(algorithm)

    def __run_algorithm(self, algorithm: AbstractAlgorithm):
        while True:
            algorithm.run(True)

    def run(self):
        for algorithm in self.algorithms[1:]:
            thread = threading.Thread(target=self.__run_algorithm, args=(algorithm, ), daemon=True)
            thread.start()

        while True:
            self.__run_algorithm(self.algorithms[0])
