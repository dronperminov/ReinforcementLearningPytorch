import abc
import pygame
from environments.abstract_environment import AbstractEnvironment


class AbstractAlgorithm:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment

        self.plot_rewards = config['plot_rewards']
        self.plot_keys = config['plot_keys']
        self.info_keys = {key: '' for key in ['episode'] + config['info_keys']}

        self.plots = dict()
        self.episode = 0

        if self.plot_rewards:
            self.plots['reward'] = []
            self.plots['reward_avg'] = []

        for key in self.plot_keys:
            self.plots[key] = []
            self.plots[f'{key}_avg'] = []

        self.x0 = config['x0']
        self.y0 = config['y0']
        self.width = config['width']
        self.height = config['height']

        self.total_width = self.width * (1 + len(self.plot_keys) + self.plot_rewards)
        self.total_height = self.height + 25 + (25 if len(self.info_keys) else 0)

        self.avg_len = config['avg_len']
        self.font_size = max(self.width // 40, 12)
        self.font = pygame.font.SysFont('Arial', self.font_size)
        self.screen = config['screen']
        self.rect = pygame.Rect(self.x0, self.y0, self.total_width, self.total_height)

    @abc.abstractmethod
    def run(self, draw: bool = True):
        pass

    @abc.abstractmethod
    def get_title(self) -> str:
        pass

    def draw(self):
        pygame.draw.rect(self.screen, (255, 255, 255), self.rect, 0)
        img = self.environment.draw(self.width, self.height)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.screen.blit(surf, (self.x0, self.y0 + 25))

        if self.plot_rewards:
            self.__plot_key(self.x0 + self.width, self.y0 + 25, 'reward')

        for i, key in enumerate(self.plot_keys):
            self.__plot_key(self.x0 + (i + 1 + self.plot_rewards) * self.width, self.y0 + 25, key)

        info = [f"{key}: {value}" for key, value in self.info_keys.items()]
        self.__draw_text(", ".join(info), self.x0 + self.total_width // 2, self.y0 + self.height + 30, 'center', 'top')
        self.__draw_text(self.get_title(), self.x0 + self.total_width // 2, self.y0 + 5, 'center', 'top')
        pygame.event.pump()
        pygame.display.update(self.rect)

    def end_episode(self, episode_reward: float, info: dict):
        if self.plot_rewards:
            self.__append_plot('reward', episode_reward)

        for key in self.plot_keys:
            self.__append_plot(key, info[key])

        for key in self.info_keys:
            if key in info:
                self.info_keys[key] = info[key]

        self.episode += 1
        self.info_keys['episode'] = f'{self.episode}'
        print(f'End episode {self.episode} with reward {episode_reward}')

    def __draw_text(self, text: str, x: int, y: int, text_align: str, text_baseline: str):
        text_surf = self.font.render(text, False, (0, 0, 0))
        text_rect = text_surf.get_rect()

        if text_align == 'right':
            x -= text_rect.right - text_rect.left
        elif text_align == 'center':
            x -= (text_rect.right - text_rect.left) // 2

        if text_baseline == 'bottom':
            y -= text_rect.bottom - text_rect.top
        elif text_baseline == 'middle':
            y -= (text_rect.bottom - text_rect.top) // 2

        self.screen.blit(text_surf, (x, y))

    def __plot_key(self, x0: int, y0: int, key: str, min_count: int = 15, max_count: int = 100):
        values = self.plots[key]
        values_avg = self.plots[f'{key}_avg']
        padding = self.font_size + 5

        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x0, y0, self.width, self.height), 0)
        pygame.draw.aalines(self.screen, (0, 0, 0), False, [
            [x0 + padding, y0 + padding],
            [x0 + padding, y0 + self.height - padding],
            [x0 + self.width - padding, y0 + self.height - padding]
        ])

        if self.episode < 2:
            return

        count = max(len(values[-max_count:]), min_count)
        min_value, max_value = min(values), max(values)

        if min_value == max_value:
            max_value += 1

        value_lines = []
        average_value_lines = []

        for i, (reward, avg_reward) in enumerate(zip(values[-max_count:], values_avg[-max_count:])):
            x = x0 + padding + i / (count - 1) * (self.width - 2 * padding)
            y = y0 + self.height - padding - ((reward - min_value) / (max_value - min_value)) * (self.height - 2 * padding)
            y_avg = y0 + self.height - padding - ((avg_reward - min_value) / (max_value - min_value)) * (self.height - 2 * padding)

            value_lines.append([x, y])
            average_value_lines.append([x, y_avg])

        pygame.draw.aalines(self.screen, (0, 150, 136), False, value_lines)
        pygame.draw.aalines(self.screen, (244, 67, 54), False, average_value_lines)

        self.__draw_text(f'{max_value:.2f}', x0 + 2, y0 + padding, 'left', 'bottom')
        self.__draw_text(f'{min_value:.2f}', x0 + 2, y0 + self.height - padding + 2, 'left', 'top')
        self.__draw_text(f'{self.episode}', value_lines[-1][0], y0 + self.height - padding + 2, 'right', 'top')
        self.__draw_text(f'{key}', x0 + self.width // 2, y0 + padding + 2, 'center', 'bottom')

    def __append_plot(self, key: str, value: float):
        self.plots[key].append(value)
        self.plots[f'{key}_avg'].append(sum(self.plots[key][-self.avg_len:]) / len(self.plots[key][-self.avg_len:]))
