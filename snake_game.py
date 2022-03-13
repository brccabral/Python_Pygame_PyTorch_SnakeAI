import math
from typing import List, Tuple
import pygame
import random
from settings import GAME_WIDTH, GAME_HEIGHT, GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, GREEN1, GREEN2, BLACK, BLUE1, BLUE2, BLOCK_SIZE, BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, WHITE, RED
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
            return Point(self.x*other, self.y * other)
        return Point(self.x * other.x, self.y * other.y)

    def __and__(self, other: "Point"):
        horizontal = self.x if other.x else 0
        vertical = self.y if other.y else 0
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
    def __init__(self, point: Point, previous: "Node", from_direction: Direction, game: "SnakeGameAI"):
        self.previous = previous
        self.point = point
        self.cost = 0
        self.next_right = None
        self.next_left = None
        self.next_down = None
        self.next_up = None

        if from_direction == Direction.RIGHT:
            self.next_left = previous
        elif from_direction == Direction.LEFT:
            self.next_right = previous
        elif from_direction == Direction.UP:
            self.next_down = previous
        elif from_direction == Direction.DOWN:
            self.next_up = previous

        if self.point.x % 2:
            self.next_down = -1
        else:
            self.next_up = -1

        if self.point.y % 2:
            self.next_right = -1
        else:
            self.next_left = -1

        if previous is not None:
            self.update_cost(game)

        if game.is_out_of_board(self.point+Direction.RIGHT):
            self.next_right = -1
        if game.is_out_of_board(self.point+Direction.LEFT):
            self.next_left = -1
        if game.is_out_of_board(self.point+Direction.DOWN):
            self.next_down = -1
        if game.is_out_of_board(self.point+Direction.UP):
            self.next_up = -1

        if game.step_cost(self.point + Direction.RIGHT, self.cost) < 0:
            self.next_right = -1
        if game.step_cost(self.point + Direction.LEFT, self.cost) < 0:
            self.next_left = -1
        if game.step_cost(self.point + Direction.DOWN, self.cost) < 0:
            self.next_down = -1
        if game.step_cost(self.point + Direction.UP, self.cost) < 0:
            self.next_up = -1

    def is_complete(self):
        return (self.next_right is not None and self.next_left is not None
                and self.next_down is not None and self.next_up is not None)

    def get_next_empty(self, game: "SnakeGameAI"):
        new_node = self
        if self.next_right is None:
            new_node = Node(self.point + Direction.RIGHT,
                            self, Direction.RIGHT, game)
            self.next_right = new_node
        elif self.next_left is None:
            new_node = Node(self.point + Direction.LEFT,
                            self, Direction.LEFT, game)
            self.next_left = new_node
        elif self.next_down is None:
            new_node = Node(self.point + Direction.DOWN,
                            self, Direction.DOWN, game)
            self.next_down = new_node
        elif self.next_up is None:
            new_node = Node(self.point + Direction.UP,
                            self, Direction.UP, game)
            self.next_up = new_node
        return new_node

    def update_cost(self, game: "SnakeGameAI"):
        if self.previous is not None:
            next_cost = self.previous.cost + 1
        else:
            next_cost = 0
        self.cost = game.step_cost(self.point, next_cost)

    def __repr__(self):
        return f"Node({self.point}, cost:{self.cost}, previous:{self.previous.point if self.previous is not None else -1})"

    def has_next(self):
        return type(self.next_down) == Node or type(self.next_left) == Node or type(self.next_right) == Node or type(self.next_up) == Node

    def get_next_node(self):
        node = None

        if type(self.next_right) == Node and self.next_right.point != self.previous.point:
            node = self.next_right
        elif type(self.next_left) == Node and self.next_left.point != self.previous.point:
            node = self.next_left
        elif type(self.next_down) == Node and self.next_down.point != self.previous.point:
            node = self.next_down
        elif type(self.next_up) == Node and self.next_up.point != self.previous.point:
            node = self.next_up

        return node

    def is_allowed_right(self):
        return self.point.y % 2 == 0

    def is_allowed_left(self):
        return self.point.y % 2

    def is_allowed_down(self):
        return self.point.x % 2 == 0

    def is_allowed_up(self):
        return self.point.y % 2

    def allowed_moves(self):
        return [self.is_allowed_down(), self.is_allowed_left(), self.is_allowed_right(), self.is_allowed_up()]

    def allowed_directions(self):
        if self.point.x % 2:
            y = -1 if self.next_up != -1 else 0
        else:
            y = 1 if self.next_down != -1 else 0
        if self.point.y % 2:
            x = -1 if self.next_left != -1 else 0
        else:
            x = 1 if self.next_right != -1 else 0
        return Point(x, y)


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
        x = random.randint(0, GAME_TABLE_COLUMNS-1)
        y = random.randint(0, GAME_TABLE_ROWS-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

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
            if self.is_collision() or self.count_steps > min(2*(GAME_TABLE_ROWS*GAME_TABLE_COLUMNS), 100*len(self.snake)):
                self.game_over = True
                reward = -10
                self.snake.pop()
                return reward, self.game_over, self.score

            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                self._place_food()
                reward = 10
                self.count_steps = 0
            else:
                self.snake.pop()

            self.traverse_table()
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

    def manhattan_distance(self, pt: Point = None) -> int:
        return abs(self.food.x - pt.x) + abs(self.food.y - pt.y)

    def is_out_of_board(self, pt: Point):
        if pt.x < 0 or pt.y < 0 or pt.x > GAME_TABLE_COLUMNS-1 or pt.y > GAME_TABLE_ROWS-1:
            return True
        return False

    def step_cost(self, pt: Point, cost: int):
        if self.is_out_of_board(pt):
            return -1
        if pt in self.snake:
            snake_index = self.snake.index(pt)
            if snake_index + cost > len(self.snake):
                return cost
            return -1
        if pt == self.food:
            cost += 1
        return cost

    def get_next_attempt(self, node: Node):
        food_direction = node.point.target_directions(self.food)
        food_horizontal = food_direction & Direction.HORIZONTAL
        food_vertical = food_direction & Direction.VERTICAL
        allowed_directions = node.allowed_directions()
        allowed_horizontal = allowed_directions & Direction.HORIZONTAL
        allowed_vertical = allowed_directions & Direction.VERTICAL

        horizontal_move = food_horizontal & allowed_horizontal
        vertical_move = food_vertical & allowed_vertical

        next_horizontal: Point = node.point + horizontal_move
        next_vertical: Point = node.point + vertical_move

        if self.traverse_path[next_horizontal.y][next_horizontal.x] != 0:
            next_horizontal_node = self.traverse_path[next_horizontal.y][next_horizontal.x]
        else:
            next_horizontal_node = Node(
                next_horizontal, node, horizontal_move, self)

        if self.traverse_path[next_vertical.y][next_vertical.x] != 0:
            next_vertical_node = self.traverse_path[next_vertical.y][next_vertical.x]
        else:
            next_vertical_node = Node(next_vertical, node, vertical_move, self)

        next_horizontal_allowed = next_horizontal_node.allowed_directions()
        next_horizontal_food = next_horizontal_node.point.target_directions(
            self.food)

        next_vertical_allowed = next_vertical_node.allowed_directions()
        next_vertical_food = next_vertical_node.point.target_directions(
            self.food)

        if next_horizontal_node.point != node.point:
            if (next_horizontal_allowed & Direction.HORIZONTAL) == (next_horizontal_food & Direction.HORIZONTAL):
                return horizontal_move
            if (next_horizontal_allowed & Direction.VERTICAL) == (next_horizontal_food & Direction.VERTICAL):
                return horizontal_move

        if next_vertical_node.point != node.point:
            if (next_vertical_allowed & Direction.VERTICAL) == (next_vertical_food & Direction.VERTICAL):
                return vertical_move
            if (next_vertical_allowed & Direction.HORIZONTAL) == (next_vertical_food & Direction.HORIZONTAL):
                return vertical_move

        if (next_horizontal_allowed & Direction.HORIZONTAL) != Point(0, 0):
            return next_horizontal_allowed & Direction.HORIZONTAL

        if (next_vertical_allowed & Direction.VERTICAL) != Point(0, 0):
            return next_vertical_allowed & Direction.VERTICAL

        return Point(0, 0)

    def get_new_node(self, node: Node):
        next_attempt = self.get_next_attempt(node)

        next_point = node.point + next_attempt
        if next_point == node.point:
            if node.is_complete():
                return node.previous
            return node

        if self.traverse_path[next_point.y][next_point.x] != 0:
            new_node = self.traverse_path[next_point.y][next_point.x]
        else:
            new_node = Node(next_point, node, next_attempt, self)

        if next_attempt == Direction.RIGHT:
            if new_node.cost < 0:
                node.next_right = -1
                return node
            node.next_right = new_node
        elif next_attempt == Direction.LEFT:
            if new_node.cost < 0:
                node.next_left = -1
                return node
            node.next_left = new_node
        elif next_attempt == Direction.DOWN:
            if new_node.cost < 0:
                node.next_down = -1
                return node
            node.next_down = new_node
        elif next_attempt == Direction.UP:
            if new_node.cost < 0:
                node.next_up = -1
                return node
            node.next_up = new_node
        else:
            new_node = node.previous

        return new_node

    def traverse_table(self):
        self.display.fill(BLACK)

        head = self.snake[0]
        # reset table
        self.traverse_path: List[List[Node]] = [
            [0 for _ in range(GAME_TABLE_COLUMNS)] for __ in range(GAME_TABLE_ROWS)]
        node = Node(head, None, None, self)
        node.cost = 0  # reset cost for head
        self.traverse_path[head.y][head.x] = node

        steps = 0
        found_food = False

        while not (node.point == head and node.is_complete()):
            if node.point == head:
                node = node.get_next_empty(self)
            else:
                node = self.get_new_node(node)

            self.traverse_path[node.point.y][node.point.x] = node
            self._display_block(WHITE, node.point)
            text = self.font_symbols.render(
                self._get_unicode(node.point), True, BLACK)
            self.display.blit(
                text, [node.point.x*BLOCK_SIZE, node.point.y*BLOCK_SIZE])
            print("*" * GAME_TABLE_COLUMNS)
            print(self.get_board())

            if node.cost > steps:
                steps = node.cost

            if node.point == self.food:
                found_food = True

            if steps >= len(self.snake) and found_food:
                steps = node.cost
                node = self.traverse_path[head.y][head.x]

    def traverse_cost(self, direction: Direction):
        print("*" * GAME_TABLE_COLUMNS)
        print(self.get_board())
        head: Point = self.snake[0]
        point: Point = head + direction
        if self.is_out_of_board(point):
            return -1
        node = self.traverse_path[point.y][point.x]
        if node == 0:
            return -1

        cost = 1
        while node.has_next():
            cost += 1
            node = node.get_next_node()

        return cost

    def update_ui(self):
        # self.display.fill(BLACK)

        self._display_block(GREEN1, self.head)
        self._display_block(GREEN2, self.head, BLOCK_DRAW_OFFSET)
        for pt in self.snake[1:]:
            self._display_block(BLUE1, pt)
            self._display_block(BLUE2, pt, BLOCK_DRAW_OFFSET)

        self._display_block(RED, self.food)

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        # pygame.display.update()

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

        self.score = 0
        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.count_steps = 0

        self.game_over = False

        self.head = Point(20, 12)
        self.snake = [self.head, self.head+Direction.DOWN,
                      self.head+Direction.DOWN+Direction.DOWN]
        self.food = Point(25, 19)
        self.direction = Direction.UP
        self.traverse_table()
        return
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
                            self.head.y-self.direction.y),
                      Point(self.head.x-self.direction.x*2,
                            self.head.y-self.direction.y*2)]

        self.food = None
        self._place_food()

    def get_board(self, marker: Point = None):
        rows = []
        for y in range(GAME_TABLE_ROWS):
            columns = []
            for x in range(GAME_TABLE_COLUMNS):
                pt = Point(x, y)
                if pt == self.food:
                    columns.append('F')
                elif pt == self.snake[0]:
                    columns.append('H')
                elif pt in self.snake[1:]:
                    columns.append('X')
                elif pt == marker:
                    columns.append('M')
                elif self.traverse_path[pt.y][pt.x] != 0:
                    columns.append(self._get_unicode(pt))
                else:
                    columns.append('_')
            rows.append("|".join(columns))
        return "\n".join(rows)

    def _get_unicode(self, pt: Point):
        if pt.x % 2 and pt.y % 2:
            return '\u25F0'
        if pt.x % 2 == 0 and pt.y % 2:
            return '\u25F1'
        if pt.x % 2 and pt.y % 2 == 0:
            return '\u25F3'
        if pt.x % 2 == 0 and pt.y % 2 == 0:
            return '\u25F2'
