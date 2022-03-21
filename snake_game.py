import datetime
import math
from typing import Tuple
import pygame
import random
from settings import GAME_WIDTH, GAME_HEIGHT, GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, GREEN1, GREEN2, BLACK, BLUE1, BLUE2, BLOCK_SIZE, BLOCK_DRAW_OFFSET, WHITE, RED
from collections import deque


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x:{self.x}, y:{self.y})"

    def __add__(self, other: "Point"):
        if type(other) != Point:
            return False
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other: "Point"):
        if type(other) != Point:
            return False
        return Point(self.x-other.x, self.y-other.y)

    def __eq__(self, other: "Point"):
        if type(other) != Point:
            return False
        return self.x == other.x and self.y == other.y

    def distance(self, other: "Point") -> int:
        if type(other) != Point:
            return False
        return abs(self.x-other.x) + abs(self.y-other.y)

    def target_directions(self, target: "Point"):
        fd = target - self
        if fd.x < 0:
            x = -1
        elif fd.x > 0:
            x = 1
        else:
            x = 0
        if fd.y < 0:
            y = -1
        elif fd.y > 0:
            y = 1
        else:
            y = 0
        return Point(x, y)

    def unit(self):
        if self.x < 0:
            x = -1
        elif self.x > 0:
            x = 1
        else:
            x = 0
        if self.y < 0:
            y = -1
        elif self.y > 0:
            y = 1
        else:
            y = 0
        return Point(x, y)

    def __mul__(self, other: "Point"):
        if type(other) == int:
            return Point(self.x * other, self.y * other)
        return Point(self.x * other.x, self.y * other.y)

    def __and__(self, other: "Point"):
        horizontal = self.x if other.x != 0 else 0
        vertical = self.y if other.y != 0 else 0
        return Point(horizontal, vertical)

    def dot(self, other: "Point"):
        mul = self * other
        return mul.x + mul.y

    def magnitude(self):
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2))

    def cos_teta(self, other: "Point"):
        if other == Point(0, 0):
            return -1
        return self.dot(other)/(self.magnitude()*other.magnitude())

    def allowed_directions(self):
        if self.x % 2:
            y = 1
        else:
            y = -1
        if self.y % 2:
            x = -1
        else:
            x = 1
        return Point(x, y)


class Direction:
    RIGHT = Point(1, 0)
    LEFT = Point(-1, 0)
    UP = Point(0, -1)
    DOWN = Point(0, 1)

    VERTICAL = Point(0, 1)
    HORIZONTAL = Point(1, 0)

    @classmethod
    def get_direction(cls, current_direction: Point, action=[0, 0, 0, 0]) -> Point:
        try:
            current_index = action.index(1)
        except ValueError:
            current_index = -1

        if current_index == 1:
            return cls.LEFT
        elif current_index == 3:
            return cls.UP
        elif current_index == 2:
            return cls.RIGHT
        elif current_index == 0:
            return cls.DOWN
        else:
            return current_direction


class SnakeGameAI:

    def __init__(self):
        # init display
        self.display = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.main_window = pygame.display.get_surface()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.font_symbols = pygame.font.SysFont("DejaVu Sans", 15)
        self.snake = deque(maxlen=GAME_TABLE_COLUMNS*GAME_TABLE_ROWS)

        self.reset()

    def _place_food(self):
        x = self.head.x
        y = self.head.y
        self.food = Point(x, y)
        while self.food in self.snake:
            x = random.randint(0, GAME_TABLE_COLUMNS-1)
            y = random.randint(0, GAME_TABLE_ROWS-1)
            self.food = Point(x, y)

    def play_step(self, action):
        """receives an action from the agent and updates the game

        Args:
            action (list): list of integers that will determine where to move the snake. [down, left, right, up]

        Returns:
            tuple: reward, game_over, score
        """
        reward = 0
        if not self.game_over:
            self.count_steps += 1

            # 2. move
            self._move(action)  # update the head

            # 3. check if game over
            if self.is_collision():
                self.game_over = True
                reward = -10
                self.snake.pop()
                return reward, self.game_over, self.score

            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                self.total_steps += self.count_steps
                print(
                    f'{datetime.datetime.now()} Score: {self.score} Steps: {self.count_steps} Total: {self.total_steps}')
                reward = 10
                self.count_steps = 0
                if len(self.snake) < GAME_TABLE_COLUMNS*GAME_TABLE_ROWS:
                    self._place_food()
                else:
                    self.game_over = True
                    return reward, self.game_over, self.score
            else:
                self.snake.pop()

            self._get_distances()
            self._create_dijkstra()
            # self.traverse_table()
            # self.towards_food()
        # 6. return game over and score
        return reward, self.game_over, self.score

    def is_collision(self, pt: Point = None):
        if pt is None:
            pt = self.head

        # hits boundary
        if self.is_out_of_board(pt):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def food_distance(self, pt: Point = None) -> int:
        if self.is_collision(pt):
            return -1
        head_distance = self.manhattan_distance(self.snake[0])
        point_distance = self.manhattan_distance(pt)
        return int((head_distance - point_distance) > 0)

    def is_out_of_board(self, pt: Point):
        if pt.x < 0 or pt.y < 0 or pt.x > GAME_TABLE_COLUMNS-1 or pt.y > GAME_TABLE_ROWS-1:
            return True
        return False

    def _get_distances(self):
        self.manhattan_distances = [
            (self.head + Direction.DOWN).distance(self.food),
            (self.head + Direction.LEFT).distance(self.food),
            (self.head + Direction.RIGHT).distance(self.food),
            (self.head + Direction.UP).distance(self.food)
        ]

    def update_ui(self):
        self.display.fill(BLACK)

        for r, row in enumerate(self.dijkstra):
            for c, column in enumerate(row):
                text = self.font_symbols.render(f'{column}', True, WHITE)
                self.display.blit(text, [c*BLOCK_SIZE, r*BLOCK_SIZE])

        self._display_block(GREEN1, self.head)
        self._display_block(GREEN2, self.head, BLOCK_DRAW_OFFSET)
        for pt in self.snake[1:]:
            self._display_block(BLUE1, pt)
            self._display_block(BLUE2, pt, BLOCK_DRAW_OFFSET)

        self._display_block(RED, self.food)

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

    def _display_block(self, color: Tuple[int, int, int], point: Point, offset: int = 0):
        pygame.draw.rect(self.display, color, pygame.Rect(
            point.x*BLOCK_SIZE + offset, point.y*BLOCK_SIZE + offset, BLOCK_SIZE-2*offset, BLOCK_SIZE-2*offset))

    def _move(self, action):
        """move snake head to new position

        Args:
            action (list): list of integers that will determine where to move the snake. [down, left, right, up]
        """
        self.direction = Direction.get_direction(self.direction, action)

        self.head = Point(self.head.x + self.direction.x,
                          self.head.y + self.direction.y)
        self.snake.insert(0, self.head)

    def reset(self):
        """reset is called at __init__ and by AI agent
        """

        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.count_steps = 0
        self.total_steps = 0

        self.game_over = False

        # init game state
        direction = random.randint(0, 3)
        if direction == 0:
            self.direction = Direction.LEFT
        elif direction == 1:
            self.direction = Direction.UP
        elif direction == 2:
            self.direction = Direction.RIGHT
        else:
            self.direction = Direction.DOWN

        self.head = Point(GAME_TABLE_COLUMNS//2, GAME_TABLE_ROWS//2)
        self.snake = [self.head,
                      Point(self.head.x-self.direction.x,
                            self.head.y-self.direction.y)]
        self.score = len(self.snake)

        self.food = None
        self._place_food()
        self._get_distances()
        self._create_dijkstra()

    def get_board(self, marker: Point = None):
        rows = []
        for y in range(GAME_TABLE_ROWS):
            columns = []
            for x in range(GAME_TABLE_COLUMNS):
                pt = Point(x, y)
                if self.dijkstra[pt.y][pt.x] != -1:
                    columns.append(f'{self.dijkstra[pt.y][pt.x]:03}')
                # if pt == self.food:
                #     columns.append('F')
                # elif pt == self.snake[0]:
                #     columns.append('H')
                # elif pt in self.snake[1:]:
                #     columns.append('X')
                # elif pt == marker:
                #     columns.append('M')
                # # elif self.traverse_path[pt.y][pt.x] != 0:
                # #     columns.append(self._get_unicode(pt))
                # else:
                #     columns.append('_')
            rows.append("|".join(columns))
        return "\n".join(rows)

    def _get_unicode(self, pt: Point):
        if pt.x % 2 == 0 and pt.y % 2:
            return '\u25F0'
        if pt.x % 2 and pt.y % 2:
            return '\u25F1'
        if pt.x % 2 == 0 and pt.y % 2 == 0:
            return '\u25F3'
        if pt.x % 2 and pt.y % 2 == 0:
            return '\u25F2'

    def is_gap(self):
        directions = self.head.allowed_directions()
        if directions.x == 0 or directions.y == 0:
            return False

        target_point = self.head + directions + self.direction
        turn_point = self.head + directions - self.direction

        if (target_point) in self.snake and not self.is_collision(turn_point):
            return directions - self.direction

        return False

    def _create_dijkstra(self):
        # the distance will never be this value, we can use it as control number
        maximum = GAME_TABLE_COLUMNS*GAME_TABLE_ROWS
        self.dijkstra = [[maximum for c in range(
            GAME_TABLE_COLUMNS)] for r in range(GAME_TABLE_ROWS)]

        # set head to 0
        self.dijkstra[self.head.y][self.head.x] = 0
        # set snake body to one less than maximum, also will never be a distance
        for s in self.snake[1:]:
            self.dijkstra[s.y][s.x] = maximum - 1
        # reset tail
        self.dijkstra[self.snake[-1].y][self.snake[-1].x] = maximum

        steps = 0
        neighbors = []
        visited = [self.head]
        while len(visited) > 0:
            current = visited.pop()
            self.dijkstra[current.y][current.x] = steps
            current_direction = current.allowed_directions()
            next_attempt_h = current + \
                (current_direction & Direction.HORIZONTAL)
            next_attempt_v = current + \
                (current_direction & Direction.VERTICAL)
            if not self.is_collision(next_attempt_h) and self.dijkstra[next_attempt_h.y][next_attempt_h.x] == maximum and next_attempt_h not in neighbors:
                neighbors.append(next_attempt_h)
            if not self.is_collision(next_attempt_v) and self.dijkstra[next_attempt_v.y][next_attempt_v.x] == maximum and next_attempt_v not in neighbors:
                neighbors.append(next_attempt_v)
            if len(visited) == 0 and len(neighbors) > 0:
                steps += 1
                visited += neighbors
                neighbors = []
