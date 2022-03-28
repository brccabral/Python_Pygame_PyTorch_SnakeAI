from __future__ import annotations
from collections import deque
import random
from typing import Deque, List, Union

from settings import GAME_TABLE_COLUMNS, GAME_TABLE_ROWS


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x:{self.x}, y:{self.y})"

    def __add__(self, other: "Point"):
        if type(other) != Point:
            raise TypeError(f'Type {type(other)} invalid')
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other: "Point"):
        if type(other) != Point:
            raise TypeError(f'Type {type(other)} invalid')
        return Point(self.x-other.x, self.y-other.y)

    def __and__(self, other: "Point"):
        if type(other) != Point:
            raise TypeError(f'Type {type(other)} invalid')
        horizontal = self.x if other.x != 0 else 0
        vertical = self.y if other.y != 0 else 0
        return Point(horizontal, vertical)

    def __eq__(self, other: "Point"):
        if type(other) != Point:
            raise TypeError(f'Type {type(other)} invalid')
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def distance(self, other: "Point") -> int:
        if type(other) != Point:
            raise TypeError(f'Type {type(other)} invalid')
        return abs(self.x-other.x) + abs(self.y-other.y)

    def target_directions(self, other: "Point"):
        if type(other) != Point:
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

    def dot(self, other: "Point"):
        if type(other) != Point and type(other) != Direction:
            raise TypeError(f'Type {type(other)} invalid')
        mul = self * other
        return mul.x + mul.y

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


class SnakeLinkedList:
    def __init__(self, max_rows: int, max_columns: int):
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.maximum = max_rows * max_columns

        self.body: Deque[Point] = deque([], maxlen=self.maximum)
        self.head: Point = None
        self.tail: Point = None

        self.food: Point = None

    def is_hit_list(self, pt: Point):
        if pt == self.head:
            return False

        for current in self.body:
            if current == pt:
                return True
        return False

    def move(self, action: List[int]):
        self.direction = Direction.get_direction(self.direction, action)
        self.body.appendleft(self.head + self.direction)
        self.head = self.body[0]
        if self.head != self.food:
            self.body.pop()
            self.tail = self.body[-1]

    def reset(self):
        # init game state
        rand_direction = random.randint(0, 3)
        if rand_direction == 0:
            self.direction = Direction.LEFT
        elif rand_direction == 1:
            self.direction = Direction.UP
        elif rand_direction == 2:
            self.direction = Direction.RIGHT
        else:
            self.direction = Direction.DOWN

        x = random.randint(1, GAME_TABLE_COLUMNS-2)
        y = random.randint(1, GAME_TABLE_ROWS-2)

        self.body.append(Point(x, y))
        self.body.append(Point(x, y) - self.direction)
        self.head = self.body[0]
        self.tail = self.body[-1]

        self.update_board()
        self.place_food()

    def __len__(self):
        return len(self.body)

    def __repr__(self):
        return f"{self.head} Length: {self._length}"

    def random_point(self):
        x = random.randint(0, self.max_columns-1)
        y = random.randint(0, self.max_rows-1)
        return Point(x, y)

    def place_food(self):
        self.food = self.random_point()
        while self.grid[self.food.y][self.food.x] == self.maximum-1 or self.head == self.food:
            self.food = self.random_point()

    def is_out_of_board(self, pt: Point):
        if pt.x < 0 or pt.y < 0 or pt.x > self.max_columns-1 or pt.y > self.max_rows-1:
            return True
        return False

    def update_board(self):
        self.grid = [[self.maximum for c in range(
            self.max_columns)] for r in range(self.max_rows)]

        for current in self.body:
            self.grid[current.y][current.x] = self.maximum - 1

        # reset head
        self.grid[self.head.y][self.head.x] = self.maximum

    def is_collision(self, pt: Point):
        if self.is_out_of_board(pt):
            return True
        if self.is_hit(pt):
            return True
        return False

    def is_hit(self, pt: Point):
        if self.grid[pt.y][pt.x] == self.maximum-1:
            return True

    def get_snake_index(self, pt: Point):
        return self.body.index(pt)
