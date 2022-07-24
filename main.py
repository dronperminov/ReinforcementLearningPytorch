import json
import argparse

from environments.abstract_environment import AbstractEnvironment
from environments.snake import Snake

from algorithms.random_algorithm import RandomAlgorithm
from algorithms.dqn import DQN
from algorithms.reinforce import Reinforce
from algorithms.actor_critic import ActorCritic
from algorithms.a2c import AdvantageActorCritic


def get_environment(config: json) -> AbstractEnvironment:
    if "environment" not in config:
        raise ValueError("Environment is not set")

    environment = config["environment"]
    environment_name = environment["name"]

    if environment_name == "snake":
        return Snake(**environment.get("config", {}))

    raise ValueError(f"Unknown environment \"{environment_name}\"")


def get_algorithm(config: json, environment: AbstractEnvironment):
    if "algorithm" not in config:
        raise ValueError("Algorithm is not set")

    algorithm = config["algorithm"]
    algorithm_name = algorithm["name"]

    if algorithm_name == "random":
        return RandomAlgorithm(environment, algorithm.get("config", {}))

    if algorithm_name == "dqn":
        return DQN(environment, algorithm.get("config", {}))

    if algorithm_name == "reinforce":
        return Reinforce(environment, algorithm.get("config", {}))

    if algorithm_name == "actor_critic":
        return ActorCritic(environment, algorithm.get("config", {}))

    if algorithm_name == "a2c":
        return AdvantageActorCritic(environment, algorithm.get("config", {}))

    raise ValueError(f"Unknown algorithm \"{algorithm_name}\"")


def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning sandbox")
    parser.add_argument("config", type=str, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    environment = get_environment(config)
    algorithm = get_algorithm(config, environment)

    while True:
        algorithm.run()


if __name__ == '__main__':
    main()
