from typing import List
import torch
import numpy as np

from environments.abstract_environment import AbstractEnvironment
from algorithms.abstract_algorithm import AbstractAlgorithm
from common.model import ActorCriticModel


class GeneralizedAdvantageEstimation(AbstractAlgorithm):
    def __init__(self, environment: AbstractEnvironment, config: dict):
        super().__init__(environment, config)

        self.gamma = config.get('gamma', 0.9)
        self.trace_decay = config.get('trace_decay', 0.99)
        self.save_model_path = config.get("save_model_path", f"gae_gamma{self.gamma}_trace_decay{self.trace_decay}.pth")

        if 'seed' in config:
            seed = config['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.agent = self.__init_agent(config['agent_architecture'])

        if 'agent_weights' in config:
            self.agent.load_state_dict(torch.load(config['agent_weights']))
            print(f'Model weights were loaded from "{config["agent_weights"]}"')

        self.optimizer = self.__init_optimizer(config['optimizer'], config['learning_rate'])
        self.loss = torch.nn.SmoothL1Loss()

        self.best_reward = float('-inf')
        self.log_probs = []
        self.rewards = []
        self.values = []

    def __init_agent(self, architecture: List[dict]) -> torch.nn.Module:
        agent = ActorCriticModel(architecture, self.environment.observation_space_shape, self.environment.action_space_shape)
        agent.to(self.device)
        agent.train()
        return agent

    def __init_optimizer(self, name: str, learning_rate: float):
        if name == 'sgd':
            return torch.optim.SGD(self.agent.parameters(), lr=learning_rate)

        if name == 'adam':
            return torch.optim.Adam(self.agent.parameters(), lr=learning_rate)

        raise ValueError(f"Unknown optimizer \"{name}\"")

    def get_action(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, value = self.agent(state)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob, value

    def get_discounted_rewards(self):
        discounted_rewards = []
        discounted_reward = 0

        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)

        discounted_rewards.reverse()
        discounted_rewards = torch.tensor(discounted_rewards, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        return discounted_rewards

    def get_advantages(self, values: torch.tensor):
        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(self.rewards), reversed(values)):
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * self.trace_decay
            next_value = v
            advantages.append(advantage)

        advantages.reverse()
        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def __update_policy(self):
        log_prob_actions = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)
        discounted_rewards = self.get_discounted_rewards()
        advantages = self.get_advantages(values)

        advantages = advantages.detach()
        discounted_rewards = discounted_rewards.detach()

        self.optimizer.zero_grad()
        policy_loss = -(advantages * log_prob_actions).sum()
        value_loss = self.loss(discounted_rewards, values).sum()
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

    def run(self, draw: bool = True):
        state = self.environment.reset()
        episode_reward = 0
        done = False
        info = {}

        while not done:
            action, log_prob, value = self.get_action(state)
            next_state, reward, done, info = self.environment.step(action)

            if draw:
                self.draw()

            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            episode_reward += reward
            state = next_state

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            torch.save(self.agent.state_dict(), self.save_model_path)
            print(f'Model was saved to "{self.save_model_path}"')

        self.__update_policy()
        self.end_episode(episode_reward, info)

    def get_title(self) -> str:
        return f'GAE (gamma: {self.gamma}, trace_decay: {self.trace_decay})'
