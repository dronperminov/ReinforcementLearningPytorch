from typing import List
import torch
import numpy as np

from environments.abstract_environment import AbstractEnvironment
from algorithms.abstract_algorithm import AbstractAlgorithm
from common.replay_buffer import ReplayBuffer
from common.prioritized_replay_buffer import PrioritizedReplayBuffer
from common.model import Model, DuelingModel
from common.optimizer_builder import OptimizerBuilder


class DQN(AbstractAlgorithm):
    def __init__(self, environment: AbstractEnvironment, config: dict):
        super().__init__(environment, config)
        self.batch_size = config.get('batch_size', 128)

        self.max_epsilon = config.get('max_epsilon', 1)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.epsilon = self.max_epsilon
        self.decay = config.get('decay', 0.004)
        self.gamma = config.get('gamma', 0.9)
        self.tau = config.get('tau', None)

        self.ddqn = config.get('ddqn', False)
        self.dueling = config.get('dueling', False)
        self.use_per = config.get('use_per', False)

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

        self.__update_target_model()
        self.model.train()
        self.target_model.eval()

        self.optimizer = OptimizerBuilder.build(config['optimizer'], config['learning_rate'], self.model.parameters())
        self.loss = torch.nn.SmoothL1Loss(reduction='none')

        self.min_replay_size = config.get('min_replay_size', 1000)
        self.max_replay_size = config.get('max_replay_size', 10000)

        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(self.max_replay_size, environment.observation_space_shape, self.device)
        else:
            self.replay_buffer = ReplayBuffer(self.max_replay_size, environment.observation_space_shape, self.device)

    def __init_agent(self, architecture: List[dict]) -> torch.nn.Module:
        if self.dueling:
            agent = DuelingModel(architecture, self.environment.observation_space_shape, self.environment.action_space_shape)
        else:
            last_layer = {'type': 'dense', 'size': self.environment.action_space_shape}
            agent = Model(architecture + [last_layer], self.environment.observation_space_shape)

        agent.to(self.device)
        return agent

    def __get_default_model_name(self) -> str:
        env_title = self.environment.get_title()
        dqn_title = f"{'soft_' if self.tau else ''}{'dueling_' if self.dueling else ''}{'d' if self.ddqn else ''}dqn{'_per' if self.use_per else ''}"
        return f"{env_title}_{dqn_title}_gamma{self.gamma}_batch_size{self.batch_size}.pth"

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            return np.random.choice(np.arange(self.environment.action_space_shape))

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions_q = self.model(state)

        return torch.argmax(actions_q).item()

    def __train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return

        if self.use_per:
            (states, actions, rewards, next_states, dones), indices, importance = self.replay_buffer.sample(self.batch_size)
            importance = importance ** (1 - self.epsilon)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            if self.ddqn:
                actions_q = self.model(next_states).max(1)[1].unsqueeze(1).long()
                labels_next = self.target_model(next_states).gather(1, actions_q)
            else:
                labels_next = self.target_model(next_states).max(1)[0].unsqueeze(1)

            labels = rewards + self.gamma * labels_next * (1 - dones)

        predicted_targets = self.model(states).gather(1, actions)
        loss = self.loss(predicted_targets, labels)

        if self.use_per:
            with torch.no_grad():
                loss *= importance
                self.replay_buffer.update(indices, torch.abs(predicted_targets - labels))

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def __update_target_model(self, is_soft: bool = False):
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
                self.__update_target_model(self.tau is not None)

        self.end_episode(episode_reward, info)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)

    def get_title(self) -> str:
        names = [
            'Soft ' if self.tau else '',
            'Dueling ' if self.dueling else '',
            'Double ' if self.ddqn else '',
            'DQN',
            ' with PER' if self.use_per else ''
        ]

        params = [
            f"gamma: {self.gamma}",
            f"batch_size: {self.batch_size}",
            f"eps: {self.epsilon:.3f}",
            f"train_period: {self.train_model_period}",
            f"update_period: {self.update_target_model_period}"
        ]

        return f"{''.join(names)} ({', '.join(params)})"
