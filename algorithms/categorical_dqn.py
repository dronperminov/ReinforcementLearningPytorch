from typing import List
import torch
import numpy as np

from environments.abstract_environment import AbstractEnvironment
from algorithms.abstract_algorithm import AbstractAlgorithm
from common.replay_buffer import ReplayBuffer
from common.model import CategoricalModel
from common.optimizer_builder import OptimizerBuilder


class CategoricalDQN(AbstractAlgorithm):
    def __init__(self, environment: AbstractEnvironment, config: dict):
        super().__init__(environment, config)
        self.batch_size = config.get('batch_size', 128)

        self.max_epsilon = config.get('max_epsilon', 1)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.epsilon = self.max_epsilon
        self.decay = config.get('decay', 0.004)
        self.gamma = config.get('gamma', 0.9)
        self.tau = config.get('tau', None)

        self.v_min = config.get('v_min', 0)
        self.v_max = config.get('v_max', 200)
        self.atom_size = config.get('atom_size', 51)
        self.delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        self.head_size = config.get('head_size', 0)

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

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size, device=self.device)
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size, device=self.device).long()
        self.offset = self.offset.unsqueeze(1).expand(self.batch_size, self.atom_size)

        self.model = self.__init_agent(config['agent_architecture'])
        self.target_model = self.__init_agent(config['agent_architecture'])

        if 'agent_weights' in config:
            self.model.load_state_dict(torch.load(config['agent_weights']))
            print(f'Agent weights were loaded from "{config["agent_weights"]}"')

        self.__update_target_model()
        self.model.train()
        self.target_model.eval()

        self.optimizer = OptimizerBuilder.build(config['optimizer'], config['learning_rate'], self.model.parameters())

        self.min_replay_size = config.get('min_replay_size', 1000)
        self.max_replay_size = config.get('max_replay_size', 10000)
        self.replay_buffer = ReplayBuffer(self.max_replay_size, environment.observation_space_shape, self.device)

    def __init_agent(self, architecture: List[dict]) -> torch.nn.Module:
        input_shape = self.environment.observation_space_shape
        output_shape = self.environment.action_space_shape
        agent = CategoricalModel(architecture, input_shape, output_shape, self.head_size, self.atom_size, self.support)
        agent.to(self.device)
        return agent

    def __get_default_model_name(self) -> str:
        env_title = self.environment.get_title()
        names = "".join([
            'soft_' if self.tau else '',
            'categorical_dqn'
        ])

        params = "_".join([
            f"gamma{self.gamma}",
            f"atom_size{self.atom_size}",
            f"vmin{self.v_min}",
            f"vmax{self.v_max}",
            f"batch_size{self.batch_size}",
            f"train_period{self.train_model_period}",
            f"update_period{self.update_target_model_period}"
        ])

        return f"{env_title}_{names}_{params}.pth"

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            return np.random.choice(np.arange(self.environment.action_space_shape))

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions_q = self.model(state)

        return torch.argmax(actions_q).item()

    def __calculate_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_action = self.target_model(next_states).argmax(1)
            next_dist = self.target_model.dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + (1 - dones) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            low = b.floor().long()
            up = b.ceil().long()

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (low + self.offset).view(-1), (next_dist * (up.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (up + self.offset).view(-1), (next_dist * (b - low.float())).view(-1))

        dist = self.model.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions.squeeze(1)])
        loss = -(proj_dist * log_p).sum(-1).mean()

        return loss

    def __train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        loss = self.__calculate_loss(states, actions, rewards, next_states, dones)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.item()

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
        loss = 0
        train_steps = 0

        while not done:
            action = self.get_action(state)
            next_state, reward, done, info = self.environment.step(action)

            if draw:
                self.draw()

            self.replay_buffer.add(state, action, reward, next_state, done)
            self.steps += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * max(self.steps - self.min_replay_size, 0))

            episode_reward += reward
            state = next_state

            if done and episode_reward > self.best_reward:
                self.best_reward = episode_reward
                torch.save(self.model.state_dict(), self.save_model_path)
                print(f'Model was saved to "{self.save_model_path}"')

            if self.steps % self.train_model_period == 0:
                loss += self.__train()
                train_steps += 1

            if self.steps % self.update_target_model_period == 0:
                self.__update_target_model(self.tau is not None)

        info['loss'] = loss / train_steps if train_steps else 0
        self.end_episode(episode_reward, info)

    def get_title(self) -> str:
        names = "".join([
            'Soft ' if self.tau else '',
            'Categorical DQN'
        ])

        params = ", ".join([
            f"gamma: {self.gamma}",
            f"atom_size: {self.atom_size}",
            f"v_min: {self.v_min}",
            f"v_max: {self.v_max}",
            f"batch_size: {self.batch_size}",
            f"eps: {self.epsilon:.3f}",
            f"train_period: {self.train_model_period}",
            f"update_period: {self.update_target_model_period}"
        ])

        return f"{names} ({params})"
