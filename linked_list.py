

import math


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


class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


class SLinkedList:
    def __init__(self):
        self.headval = None
