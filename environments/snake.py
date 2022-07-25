from typing import Tuple
import numpy as np
import cv2
from environments.abstract_environment import AbstractEnvironment


class Snake(AbstractEnvironment):
    INITIAL_LENGTH = 3

    TURN_LEFT = 1
    TURN_RIGHT = 2

    EAT_SELF = 'eat self'
    WALL = 'wall'
    EAT_FOOD = 'eat food'
    NO_EAT = 'no eat'
    DEFAULT = 'default'

    HEAD_CELL = 0
    SNAKE_CELL = 1
    FOOD_CELL = 2

    def __init__(self, field_width: int = 15, field_height: int = 10, use_conv: bool = False):
        self.field_width = field_width
        self.field_height = field_height
        self.use_conv = use_conv
        super().__init__(self.field_width / self.field_height)

        self.screen = None
        self.snake = None
        self.food = None
        self.direction = None
        self.steps_without_food = 0

        self.action_space_shape = 3
        self.observation_space_shape = (3, self.field_height, self.field_width) if self.use_conv else 43
        self.info = {
            'length': Snake.INITIAL_LENGTH,
            'max_length': Snake.INITIAL_LENGTH,
            'wall': 0,
            'eat_self': 0,
            'no_eat': 0
        }

    def reset(self) -> np.ndarray:
        self.snake = self.__init_snake()
        self.food = self.__init_food()
        self.direction = {'dx': 0, 'dy': -1}
        self.steps_without_food = 0
        self.info['length'] = Snake.INITIAL_LENGTH

        return self.__get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        dx, dy = self.__get_direction(action)
        self.direction['dx'], self.direction['dy'] = dx, dy

        move = self.__move_snake(dx, dy)
        done = move in [Snake.WALL, Snake.EAT_SELF, Snake.NO_EAT]
        state = self.__get_state()
        reward = self.__get_reward(move)

        return state, reward, done, self.info

    def draw(self, width: int, height: int) -> np.ndarray:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cell_width = (width - 1) // self.field_width
        cell_height = (height - 1) // self.field_height

        for p in self.snake:
            color = (0, 150, 136) if p == self.snake[0] else (76, 175, 80)
            self.__draw_cell(img, p['x'], p['y'], cell_width, cell_height, color, True)

        self.__draw_cell(img, self.food['x'], self.food['y'], cell_width, cell_height, (244, 67, 54), True)

        for i in range(self.field_width):
            for j in range(self.field_height):
                self.__draw_cell(img, i, j, cell_width, cell_height, (204, 204, 204))

        return img

    def __init_snake(self):
        snake = []
        x0 = self.field_width // 2
        y0 = self.field_height // 2

        for i in range(Snake.INITIAL_LENGTH):
            snake.append({'x': x0, 'y': y0 + i})

        return snake

    def __init_food(self):
        food = {'x': 0, 'y': 0}

        while True:
            food['x'] = np.random.randint(self.field_width)
            food['y'] = np.random.randint(self.field_height)

            if not self.__is_inside_snake(food['x'], food['y']):
                return food

    def __is_inside_snake(self, x, y, start: int = 0) -> bool:
        for p in self.snake[start:]:
            if p['x'] == x and p['y'] == y:
                return True

        return False

    def __move_snake(self, dx, dy) -> str:
        head = self.snake[0]
        head_x, head_y = head['x'], head['y']

        if head_x + dx < 0 or head_y + dy < 0 or head_x + dx >= self.field_width or head_y + dy >= self.field_height:
            self.info['wall'] += 1
            return Snake.WALL

        if self.__is_inside_snake(head_x + dx, head_y + dy, 1):
            self.info['eat_self'] += 1
            return Snake.EAT_SELF

        if head_x + dx == self.food['x'] and head_y + dy == self.food['y']:
            self.snake.insert(0, {'x': head_x + dx, 'y': head_y + dy})
            self.food = self.__init_food()
            self.info['max_length'] = max(len(self.snake), self.info['max_length'])
            self.steps_without_food = 0
            return Snake.EAT_FOOD

        self.steps_without_food += 1

        if self.steps_without_food > self.field_width * self.field_height * 2:
            self.info['no_eat'] += 1
            return Snake.NO_EAT

        for i in reversed(range(1, len(self.snake))):
            self.snake[i]['x'] = self.snake[i - 1]['x']
            self.snake[i]['y'] = self.snake[i - 1]['y']

        self.snake[0]['x'] += dx
        self.snake[0]['y'] += dy
        self.info['length'] = len(self.snake)

        return Snake.DEFAULT

    def __is_collision(self, point) -> bool:
        if point['x'] < 0 or point['x'] >= self.field_width:
            return True

        if point['y'] < 0 or point['y'] >= self.field_height:
            return True

        return self.__is_inside_snake(point['x'], point['y'])

    def __distance_to_collision(self, x0, y0, dx, dy, from_head: bool = False):
        x = x0 + (dx if from_head else 0)
        y = y0 + (dy if from_head else 0)
        i = 1

        while 0 <= x < self.field_width and 0 <= y < self.field_height and not self.__is_inside_snake(x, y, 1):
            x += dx
            y += dy
            i += 1

        return [dx * i / (self.field_width - 1), dy * i / (self.field_height - 1)]

    def __state_to_tensor(self):
        state = np.zeros(self.observation_space_shape)

        for i in range(self.field_width):
            state[:, 0, i] = -1
            state[:, self.field_height - 1, i] = -1

        for i in range(self.field_height):
            state[:, i, 0] = -1
            state[:, i, self.field_width - 1] = -1

        state[Snake.FOOD_CELL, self.food['y'], self.food['x']] = 1

        head_x, head_y = self.snake[0]['x'], self.snake[0]['y']
        state[Snake.HEAD_CELL, head_y, head_x] = 1
        dx, dy = self.direction['dx'], self.direction['dy']

        if 0 <= head_x + dx < self.field_width and 0 <= head_y + dy < self.field_height:
            state[Snake.HEAD_CELL, head_y + dy, head_x + dx] = 0.5

        if 0 <= head_x + dy < self.field_width and 0 <= head_y - dx < self.field_height:
            state[Snake.HEAD_CELL, head_y - dx, head_x + dy] = 0.5

        if 0 <= head_x - dy < self.field_width and 0 <= head_y + dx < self.field_height:
            state[Snake.HEAD_CELL, head_y + dx, head_x - dy] = 0.5

        for cell in self.snake:
            state[Snake.SNAKE_CELL, cell['y'], cell['x']] = 1

        return state

    def __state_to_vector(self) -> np.ndarray:
        head_x = self.snake[0]['x']
        head_y = self.snake[0]['y']

        food_x = self.food['x']
        food_y = self.food['y']

        point_l = {'x': head_x - 1, 'y': head_y}
        point_r = {'x': head_x + 1, 'y': head_y}
        point_u = {'x': head_x, 'y': head_y - 1}
        point_d = {'x': head_x, 'y': head_y + 1}

        dir_l = self.direction['dx'] == -1
        dir_r = self.direction['dx'] == 1
        dir_u = self.direction['dy'] == -1
        dir_d = self.direction['dy'] == 1

        distances = [
            *self.__distance_to_collision(head_x, head_y, self.direction['dx'], self.direction['dy'], True),
            *self.__distance_to_collision(head_x - self.direction['dy'], head_y + self.direction['dx'], self.direction['dx'], self.direction['dy']),
            *self.__distance_to_collision(head_x + self.direction['dy'], head_y - self.direction['dx'], self.direction['dx'], self.direction['dy']),

            *self.__distance_to_collision(head_x, head_y, -self.direction['dx'], -self.direction['dy'], True),
            *self.__distance_to_collision(head_x - self.direction['dy'], head_y + self.direction['dx'], -self.direction['dx'], -self.direction['dy']),
            *self.__distance_to_collision(head_x + self.direction['dy'], head_y - self.direction['dx'], -self.direction['dx'], -self.direction['dy']),

            *self.__distance_to_collision(head_x, head_y, self.direction['dy'], -self.direction['dx'], True),
            *self.__distance_to_collision(head_x + self.direction['dx'], head_y + self.direction['dy'], self.direction['dy'], -self.direction['dx']),
            *self.__distance_to_collision(head_x - self.direction['dx'], head_y - self.direction['dy'], self.direction['dy'], -self.direction['dx']),

            *self.__distance_to_collision(head_x, head_y, -self.direction['dy'], self.direction['dx'], True),
            *self.__distance_to_collision(head_x + self.direction['dx'], head_y + self.direction['dy'], -self.direction['dy'], self.direction['dx']),
            *self.__distance_to_collision(head_x - self.direction['dx'], head_y - self.direction['dy'], -self.direction['dy'], self.direction['dx'])
        ]

        vector = [
            (dir_u and self.__is_collision(point_u)) or (dir_d and self.__is_collision(point_d)) or
            (dir_l and self.__is_collision(point_l)) or (dir_r and self.__is_collision(point_r)),

            (dir_u and self.__is_collision(point_r)) or (dir_d and self.__is_collision(point_l)) or
            (dir_u and self.__is_collision(point_u)) or (dir_d and self.__is_collision(point_d)),

            (dir_u and self.__is_collision(point_r)) or (dir_d and self.__is_collision(point_l)) or
            (dir_r and self.__is_collision(point_u)) or (dir_l and self.__is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,
            food_x < head_x, food_x > head_x, food_y < head_y, food_y > head_y,
            self.direction['dx'], self.direction['dy'],

            (head_x - 0) / (self.field_width - 1), (head_y - 0) / (self.field_height - 1),
            (head_x - (self.field_width - 1)) / (self.field_width - 1), (head_y - (self.field_height - 1)) / (self.field_height - 1),
            (head_x - food_x) / (self.field_width - 1), (head_y - food_y) / (self.field_height - 1),
            *distances
        ]

        return np.array(vector)

    def __get_state(self):
        if self.use_conv:
            return self.__state_to_tensor()

        return self.__state_to_vector()

    def __get_reward(self, move: str) -> float:
        if move == Snake.WALL:
            return -1

        if move == Snake.EAT_SELF:
            return -2

        if move == Snake.NO_EAT:
            return -4

        if move == Snake.EAT_FOOD:
            return len(self.snake)

        head = self.snake[0]
        prev_dx = head['x'] - self.direction['dx'] - self.food['x']
        prev_dy = head['y'] - self.direction['dy'] - self.food['y']
        prev_dst = abs(prev_dx) + abs(prev_dy)

        curr_dx = head['x'] - self.food['x']
        curr_dy = head['y'] - self.food['y']
        curr_dst = abs(curr_dx) + abs(curr_dy)

        if curr_dst < prev_dst:
            return 0.5 / len(self.snake)

        return -1 / len(self.snake)

    def __get_direction(self, action: int):
        dx, dy = self.direction['dx'], self.direction['dy']

        if action == Snake.TURN_LEFT:
            return dy, -dx

        if action == Snake.TURN_RIGHT:
            return -dy, dx

        return dx, dy

    def __draw_cell(self, img: np.ndarray, x: int, y: int, cell_width: int, cell_height: int, color, filled: bool = False):
        x1, y1 = x * cell_width, y * cell_height
        x2, y2 = x1 + cell_width, y1 + cell_height
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1 if filled else 1)
