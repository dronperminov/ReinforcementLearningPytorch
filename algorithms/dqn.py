from typing import List
import torch
import numpy as np

from environments.abstract_environment import AbstractEnvironment
from algorithms.abstract_algorithm import AbstractAlgorithm
from common.replay_buffer import ReplayBuffer
from common.model import Model


class DQN(AbstractAlgorithm):
    def __init__(self, environment: AbstractEnvironment, config: dict):
        super().__init__(environment, config)
        self.batch_size = config.get('batch_size', 128)
        self.min_replay_size = config.get('min_replay_size', 1000)
        self.replay_buffer = ReplayBuffer(config.get('max_replay_size', 10000))

        self.max_epsilon = config.get('max_epsilon', 1)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.epsilon = self.max_epsilon
        self.decay = config.get('decay', 0.004)
        self.gamma = config.get('gamma', 0.9)
        self.tau = config.get('tau', None)

        self.ddqn = config.get('ddqn', False)

        self.train_model_period = config.get('train_model_period', 4)
        self.update_target_model_period = config.get('update_target_model_period', 500)
        self.save_model_path = config.get('save_model_path', self.__get_default_model_name())

        self.steps = 0
        self.best_reward = float('-inf')

        if 'seed' in config:
            seed = config['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.model = self.__init_agent(config['agent_architecture'])
        self.target_model = self.__init_agent(config['agent_architecture'])

        if 'agent_weights' in config:
            self.model.load_state_dict(torch.load(config['agent_weights']))
            print(f'Agent weights were loaded from "{config["agent_weights"]}"')

        self.__update_target_model(is_soft=False)
        self.model.train()
        self.target_model.eval()

        self.optimizer = self.__init_optimizer(config['optimizer'], config['learning_rate'])
        self.loss = torch.nn.SmoothL1Loss()

    def __init_agent(self, architecture: List[dict]) -> torch.nn.Module:
        last_layer = {'type': 'dense', 'size': self.environment.action_space_shape}
        agent = Model(architecture + [last_layer], self.environment.observation_space_shape)
        agent.to(self.device)
        return agent

    def __init_optimizer(self, name: str, learning_rate: float):
        if name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        if name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        raise ValueError(f"Unknown optimizer \"{name}\"")

    def __get_default_model_name(self) -> str:
        return f"{'soft_' if self.tau else ''}{'d' if self.ddqn else ''}dqn_gamma{self.gamma}_batch_size{self.batch_size}.pth"

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            return np.random.choice(np.arange(self.environment.action_space_shape))

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            actions_q = self.model(state)

        return torch.argmax(actions_q).item()

    def __train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)
        self.model.eval()

        with torch.no_grad():
            if self.ddqn:
                actions_q = self.model(next_states).detach().max(1)[1].unsqueeze(1).long()
                labels_next = self.target_model(next_states).gather(1, actions_q)
            else:
                labels_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)

        self.model.train()
        predicted_targets = self.model(states).gather(1, actions)
        labels = rewards + self.gamma * labels_next * (1 - dones)
        loss = self.loss(predicted_targets, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tau:
            self.__update_target_model(is_soft=True)

    def __update_target_model(self, is_soft: bool):
        if is_soft:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def run(self, draw: bool = True):
        state = self.environment.reset()
        episode_reward = 0
        done = False
        info = {}

        while not done:
            action = self.get_action(state)
            next_state, reward, done, info = self.environment.step(action)

            if draw:
                self.draw()

            self.replay_buffer.add(state, action, reward, next_state, done)
            self.steps += 1
            episode_reward += reward
            state = next_state

            if done and episode_reward > self.best_reward:
                self.best_reward = episode_reward
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f'Model was saved to "{self.save_model_path}"')

            if self.steps % self.train_model_period == 0:
                self.__train()

            if self.steps % self.update_target_model_period == 0:
                self.__update_target_model(is_soft=False)

        self.end_episode(episode_reward, info)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)

    def get_title(self) -> str:
        name = f"{'Soft ' if self.tau else ''}{'D' if self.ddqn else ''}DQN"
        params = [
            f"gamma: {self.gamma}",
            f"batch_size: {self.batch_size}",
            f"eps: {self.epsilon:.3f}",
            f"train_period: {self.train_model_period}",
        ]

        if self.tau is None:
            params.append(f"update_period: {self.update_target_model_period}")

        return f"{name} ({', '.join(params)})"
