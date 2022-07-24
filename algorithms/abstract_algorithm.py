import abc
import pygame
from environments.abstract_environment import AbstractEnvironment


class AbstractAlgorithm:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment
        self.avg_len = config.get('avg_len', 50)
        self.plot_rewards = config.get('plot_rewards', False)
        self.plot_keys = config.get('plot_keys', [])
        self.plots = dict()
        self.episode = 0

        if self.plot_rewards:
            self.plots['reward'] = []
            self.plots['reward_avg'] = []

        for key in self.plot_keys:
            self.plots[key] = []
            self.plots[f'{key}_avg'] = []

        self.width = config.get('width', 500)
        self.height = config.get('height', int(self.width / self.environment.aspect_ratio))
        self.total_width = self.width * (1 + len(self.plots) // 2)

        pygame.init()
        self.screen = pygame.display.set_mode((self.total_width, self.height))
        self.font_size = min(self.width // 40, 12)
        self.font = pygame.font.SysFont('Arial', self.font_size)

    @abc.abstractmethod
    def run(self, draw: bool = True):
        pass

    def draw(self):
        img = self.environment.draw(self.width, self.height)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        if self.plot_rewards:
            self.__plot_key(self.width, 0, 'reward')

        for i, key in enumerate(self.plot_keys):
            self.__plot_key((i + 1 + self.plot_rewards) * self.width, 0, key)

        pygame.event.pump()
        pygame.display.update()

    def end_episode(self, episode_reward: float, info: dict):
        if self.plot_rewards:
            self.__append_plot('reward', episode_reward)

        for key in self.plot_keys:
            self.__append_plot(key, info[key])

        self.episode += 1
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

    def __plot_key(self, x0: int, y0: int, key: str, min_count: int = 15):
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

        count = max(self.episode, min_count)
        min_value, max_value = min(values), max(values)

        if min_value == max_value:
            max_value += 1

        reward_lines = []
        average_reward_lines = []

        for i, (reward, avg_reward) in enumerate(zip(values, values_avg)):
            x = x0 + padding + i / (count - 1) * (self.width - 2 * padding)
            y = y0 + self.height - padding - ((reward - min_value) / (max_value - min_value)) * (self.height - 2 * padding)
            y_avg = y0 + self.height - padding - ((avg_reward - min_value) / (max_value - min_value)) * (self.height - 2 * padding)

            reward_lines.append([x, y])
            average_reward_lines.append([x, y_avg])

        pygame.draw.aalines(self.screen, (0, 150, 136), False, reward_lines)
        pygame.draw.aalines(self.screen, (244, 67, 54), False, average_reward_lines)

        self.__draw_text(f'{max_value:.2f}', x0 + 2, y0 + padding, 'left', 'bottom')
        self.__draw_text(f'{min_value:.2f}', x0 + 2, y0 + self.height - padding + 2, 'left', 'top')
        self.__draw_text(f'{self.episode}', reward_lines[-1][0], y0 + self.height - padding + 2, 'right', 'top')
        self.__draw_text(f'{key}', x0 + self.width // 2, y0 + padding + 2, 'center', 'bottom')

    def __append_plot(self, key: str, value: float):
        self.plots[key].append(value)
        self.plots[f'{key}_avg'].append(sum(self.plots[key][-self.avg_len:]) / len(self.plots[key][-self.avg_len:]))
