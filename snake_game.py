import datetime
import itertools
import math
import re
from typing import Deque, List, Tuple
import pygame
import random
from settings import GAME_WIDTH, GAME_HEIGHT, GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, BLACK, BLOCK_SIZE, BLOCK_DRAW_OFFSET, WHITE, RED
from collections import deque


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x:{self.x}, y:{self.y})"

    def __add__(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        return Point(self.x-other.x, self.y-other.y)

    def __eq__(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def distance(self, other: "Point") -> int:
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        return abs(self.x-other.x) + abs(self.y-other.y)

    def target_directions(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        fd = other - self
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
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        horizontal = self.x if other.x != 0 else 0
        vertical = self.y if other.y != 0 else 0
        return Point(horizontal, vertical)

    def dot(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        mul = self * other
        return mul.x + mul.y

    def magnitude(self):
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2))

    def cos_teta(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
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


class Snake:
    def __init__(self):
        self.reset()

    def __len__(self):
        return len(self.body)

    def move(self, action: List[int], food: Point):
        """move snake head to new position

        Args:
            action (list): list of integers that will determine where to move the snake. [down, left, right, up]
        """
        self.direction = Direction.get_direction(self.direction, action)
        self.head = self.head + self.direction
        self.body.insert(0, self.head)
        if self.head != food:
            self.body.pop()
        self.tail = self.body[-1]

    def reset(self):
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

        x = random.randint(1, GAME_TABLE_COLUMNS-2)
        y = random.randint(1, GAME_TABLE_ROWS-2)
        self.head = Point(x, y)
        self.body = []
        self.body.append(self.head)
        self.body.append(self.head-self.direction)
        self.tail = self.body[-1]


class Board:
    def __init__(self, max_rows: int, max_columns: int, snake: Snake):
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.maximum = max_rows * max_columns
        self.reset(snake)

    def random_point(self):
        x = random.randint(0, self.max_columns-1)
        y = random.randint(0, self.max_rows-1)
        return Point(x, y)

    def place_food(self, snake: Snake):
        self.food = self.random_point()
        while self.grid[self.food.y][self.food.x] == self.maximum-1 or self.food == snake.head:
            self.food = self.random_point()

    def is_out_of_board(self, pt: Point):
        if pt.x < 0 or pt.y < 0 or pt.x > self.max_columns-1 or pt.y > self.max_rows-1:
            return True
        return False

    def update_board(self, snake: Snake):
        self.grid = [[self.maximum - 1 if Point(c, r) in snake.body else self.maximum for c in range(
            self.max_columns)] for r in range(self.max_rows)]
        self.grid[snake.head.y][snake.head.x] = self.maximum

    def reset(self, snake: Snake):
        self.update_board(snake)
        self.place_food(snake)

    def is_collision(self, pt: Point):
        if self.is_out_of_board(pt):
            return True
        if self.is_hit(pt):
            return True
        return False

    def is_hit(self, pt: Point):
        if self.grid[pt.y][pt.x] == self.maximum-1:
            return True


class SnakeGameAI:

    def __init__(self):
        # init display
        self.display = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.main_window = pygame.display.get_surface()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.font_symbols = pygame.font.SysFont("DejaVu Sans", 12)
        self.turns = [Direction.DOWN,
                      Direction.LEFT, Direction.RIGHT, Direction.UP]
        self.snake = Snake()
        self.board = Board(GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, self.snake)

        self.reset()

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
            self.snake.move(action, self.board.food)  # update the head
            self.board.update_board(self.snake)

            # 3. check if game over
            if self.board.is_collision(self.snake.head):
                self.game_over = True
                reward = -10
                return reward, self.game_over, self.score

            # 4. place new food or just move
            if self.snake.head == self.board.food:
                self.score += 1
                self.total_steps += self.count_steps
                print(
                    f'{datetime.datetime.now()} Score: {self.score} Steps: {self.count_steps} Total: {self.total_steps}')
                reward = 10
                self.count_steps = 0
                if len(self.snake) < GAME_TABLE_COLUMNS*GAME_TABLE_ROWS:
                    self.board.place_food(self.snake)
                else:
                    self.game_over = True
                    return reward, self.game_over, self.score

            self._get_distances()
            self._create_dijkstra()

        # 6. return game over and score
        return reward, self.game_over, self.score

    def _get_distances(self):
        self.manhattan_distances = [
            (self.snake.head + Direction.DOWN).distance(self.board.food),
            (self.snake.head + Direction.LEFT).distance(self.board.food),
            (self.snake.head + Direction.RIGHT).distance(self.board.food),
            (self.snake.head + Direction.UP).distance(self.board.food)
        ]

    def update_ui(self):
        maximum = GAME_TABLE_COLUMNS * GAME_TABLE_ROWS
        self.display.fill(BLACK)

        for index, pt in enumerate(self.snake.body):
            index_percent = index/(len(self.snake)-1)
            blue = (1-index_percent)*255
            green = (index_percent)*255
            color = (0, green, blue)
            color2 = (0, 128, blue)
            self._display_block(color, pt)
            self._display_block(color2, pt, BLOCK_DRAW_OFFSET)

        self._display_block(RED, self.board.food)

        for r, row in enumerate(self.dijkstra):
            for c, value in enumerate(row):
                pt = Point(c, r)
                if self.board.is_collision(pt):
                    value = self.snake.body.index(pt)
                if value < maximum - 1:
                    text = self.font_symbols.render(f'{value}', True, WHITE)
                    self.display.blit(text, [pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE])

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

    def _display_block(self, color: Tuple[int, int, int], point: Point, offset: int = 0):
        pygame.draw.rect(self.display, color, pygame.Rect(
            point.x*BLOCK_SIZE + offset, point.y*BLOCK_SIZE + offset, BLOCK_SIZE-2*offset, BLOCK_SIZE-2*offset))

    def reset(self):
        """reset is called at __init__ and by AI agent
        """

        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.count_steps = 0
        self.total_steps = 0

        self.game_over = False

        self.snake.reset()

        self.board.reset(self.snake)

        self.score = len(self.snake)

        self._get_distances()
        self._create_dijkstra()

    def get_board(self, marker: Point = Point(-1, -1)):
        rows = []
        for y in range(GAME_TABLE_ROWS):
            columns = []
            for x in range(GAME_TABLE_COLUMNS):
                pt = Point(x, y)
                if pt == marker:
                    columns.append('MMM')
                elif pt == self.board.food:
                    columns.append('FFF')
                elif pt == self.snake.head:
                    columns.append('HHH')
                elif self.board.is_collision(pt):
                    columns.append('SSS')
                elif self.dijkstra[pt.y][pt.x] != -1:
                    columns.append(f'{self.dijkstra[pt.y][pt.x]:03}')

            rows.append("|".join(columns))
        return "\n".join(rows)

    def _get_unicode(self, pt: Point):
        pt = pt.unit()
        if pt.x % 2 == 0 and pt.y % 2:
            code = '\u25F0'
        elif pt.x % 2 and pt.y % 2:
            code = '\u25F1'
        elif pt.x % 2 == 0 and pt.y % 2 == 0:
            code = '\u25F3'
        elif pt.x % 2 and pt.y % 2 == 0:
            code = '\u25F2'
        else:
            code = ' '
        return code

    def is_gap(self):
        directions = self.snake.head.allowed_directions()
        if directions.x == 0 or directions.y == 0:
            return Point(0, 0)

        target_point = self.snake.head + directions + self.snake.direction
        turn_point = self.snake.head + directions - self.snake.direction

        if (target_point) in self.snake.body and not self.board.is_collision(turn_point):
            return directions - self.snake.direction

        return Point(0, 0)

    def _create_dijkstra(self):
        # the distance will never be this value, we can use it as control number
        maximum = self.board.maximum
        # set snake body to one less than maximum, also will never be a distance
        self.dijkstra = [[maximum-1 if self.board.is_collision(Point(c, r)) else maximum for c in range(
            self.board.max_columns)] for r in range(self.board.max_rows)]

        # set head to 0
        self.dijkstra[self.snake.head.y][self.snake.head.x] = 0
        # reset tail
        self.dijkstra[self.snake.tail.y][self.snake.tail.x] = maximum

        steps = 0
        neighbors = []
        working = [self.snake.head]
        # found_food = False
        # found_tail = False
        while len(working) > 0:
            working = sorted(
                working, key=lambda pt: pt.distance(self.board.food), reverse=True)
            current = working.pop()
            self.dijkstra[current.y][current.x] = steps
            if current == self.board.food:
                # found_food = True
                break
            # if current == self.snake.tail:
            #     found_tail = True
            # if found_food and found_tail:
            #     break
            current_direction = current.allowed_directions()
            next_attempt_h = current + \
                (current_direction & Direction.HORIZONTAL)
            next_attempt_v = current + \
                (current_direction & Direction.VERTICAL)
            if not self.board.is_out_of_board(next_attempt_h) and self.dijkstra[next_attempt_h.y][next_attempt_h.x] == maximum:
                self.dijkstra[next_attempt_h.y][next_attempt_h.x] = steps + 1
                neighbors.append(next_attempt_h)
            if not self.board.is_out_of_board(next_attempt_v) and self.dijkstra[next_attempt_v.y][next_attempt_v.x] == maximum:
                self.dijkstra[next_attempt_v.y][next_attempt_v.x] = steps + 1
                neighbors.append(next_attempt_v)

            if len(working) == 0 and len(neighbors) > 0:
                steps += 1
                working += neighbors
                neighbors = []
        self.shortest_dijkstra()
        self.dijkstra_gap()

    def shortest_dijkstra(self):
        maximum = GAME_TABLE_COLUMNS*GAME_TABLE_ROWS

        # target can be the food or the tail
        target = self.board.food
        if self.dijkstra[target.y][target.x] == maximum:
            target = self.snake.tail

        while self.dijkstra[target.y][target.x] != 1:
            moves = [target + turn for turn in self.turns]
            moves = sorted(moves, key=lambda pt: pt.distance(self.snake.head))
            for move in moves:
                if not self.board.is_out_of_board(move):
                    value = self.dijkstra[move.y][move.x]
                    if value == self.dijkstra[target.y][target.x] - 1:
                        break
            target = move

        self.short_dijkstra = [
            int((self.snake.head+turn) == target) for turn in self.turns]

    def dijkstra_gap(self):

        # if can't find shortest path to target
        if sum(self.short_dijkstra) == 0:
            return

        # get the suggested turn
        suggested_turn = self.turns[self.short_dijkstra.index(1)]
        control = self.snake.head + suggested_turn

        # get the other possible turn
        allowed_directions = self.snake.head.allowed_directions()
        other_turn = allowed_directions - suggested_turn

        # start at other turn
        current = self.snake.head + other_turn
        # check if other turn is collision
        if self.board.is_collision(current):
            return

        food_dijkstra = 0
        if self.dijkstra[self.board.food.y][self.board.food.x] < GAME_TABLE_COLUMNS*GAME_TABLE_ROWS - 1:
            food_dijkstra = self.dijkstra[self.board.food.y][self.board.food.x]

        # start at other turn
        working = [current]
        visited = [current]
        found_tail = False
        highest_snake_index = 0
        while len(working) > 0:
            working = sorted(
                working, key=lambda pt: pt.distance(self.snake.tail), reverse=True)

            current = working.pop()
            if current == self.snake.tail:
                found_tail = True
                break

            for turn in self.turns:
                move = current + turn

                if (not self.board.is_collision(move) or move == self.snake.tail) and move not in visited and move != control and move != self.snake.head:
                    working.append(move)
                    visited.append(move)
                elif not self.board.is_out_of_board(move) and self.board.is_hit(move):
                    snake_index = self.snake.body.index(move)
                    if snake_index > highest_snake_index:
                        highest_snake_index = snake_index

            if len(visited) > len(self.snake):
                found_tail = True
                break

        # if the tail was not found, but the steps to get to the food is greater
        # than the number of steps the tail has to do to get there, it is not a gap
        if not found_tail and food_dijkstra > len(self.snake) - highest_snake_index:
            found_tail = True

        # if it can't reach the tail, it is a gap
        if not found_tail:
            self.short_dijkstra = [
                1 if other_turn == Direction.DOWN else 0,
                1 if other_turn == Direction.LEFT else 0,
                1 if other_turn == Direction.RIGHT else 0,
                1 if other_turn == Direction.UP else 0,
            ]
