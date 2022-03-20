import datetime
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
            return Point(self.x * other, self.y * other)
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
            self.next_up = -1
        else:
            self.next_down = -1

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

    def get_next_side(self, food: Point):
        next_side = None
        food_direction = (food - self.point).unit()
        if self.next_right is None and food_direction.x > 0:
            next_side = Direction.RIGHT
        elif self.next_left is None and food_direction.x < 0:
            next_side = Direction.LEFT
        elif self.next_down is None and food_direction.y > 0:
            next_side = Direction.DOWN
        elif self.next_up is None and food_direction.y < 0:
            next_side = Direction.UP
        if next_side is not None:
            return next_side

        if self.next_right is None:
            next_side = Direction.RIGHT
        elif self.next_left is None:
            next_side = Direction.LEFT
        elif self.next_down is None:
            next_side = Direction.DOWN
        elif self.next_up is None:
            next_side = Direction.UP
        return next_side

    def update_cost(self, game: "SnakeGameAI"):
        if self.previous is not None:
            next_cost = self.previous.cost + 1
        else:
            next_cost = 0
        self.cost = game.step_cost(self.point, next_cost)

    def __repr__(self):
        return f"Node({self.point}, cost:{self.cost}, previous:{self.previous.point if self.previous is not None else -1})"


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

    def _get_distances(self):
        self.manhattan_distances = [
            (self.head + Direction.DOWN).distance(self.food),
            (self.head + Direction.LEFT).distance(self.food),
            (self.head + Direction.RIGHT).distance(self.food),
            (self.head + Direction.UP).distance(self.food)
        ]

    def traverse_table(self):
        head = self.snake[0]
        head_node = Node(head, None, None, self)
        head_node.cost = 0  # reset cost for head
        self.costs = [-1, -1, -1, -1]

        while not head_node.is_complete():
            steps = 0
            found_food = False
            # reset table
            self.traverse_path: List[List[Node]] = [
                [0 for _ in range(GAME_TABLE_COLUMNS)] for __ in range(GAME_TABLE_ROWS)]
            self.traverse_path[head.y][head.x] = head_node

            # get next side
            current_side = head_node.get_next_side(
                self.food)  # safe because head is not complete
            current_node = Node(head + current_side,
                                head_node, current_side, self)
            if current_side == Direction.DOWN:
                head_node.next_down = current_node
            elif current_side == Direction.LEFT:
                head_node.next_left = current_node
            elif current_side == Direction.RIGHT:
                head_node.next_right = current_node
            elif current_side == Direction.UP:
                head_node.next_up = current_node

            node = current_node

            while True:
                self._display_block(WHITE, node.point)
                steps += 1

                self.traverse_path[node.point.y][node.point.x] = node
                text = self.font_symbols.render(
                    self._get_unicode(node.point), True, BLACK)
                self.display.blit(
                    text, [node.point.x*BLOCK_SIZE, node.point.y*BLOCK_SIZE])
                # print("*" * GAME_TABLE_COLUMNS)
                # print(self.get_board())

                if node.point == self.food:
                    steps += -1
                    found_food = True
                if steps >= len(self.snake) and found_food:
                    break

                next_side = node.get_next_side(self.food)
                if next_side is None:
                    node = node.previous
                    steps += -1
                    if node is None:
                        break
                    if node.point == current_node.point:
                        break
                    if node.point == head:
                        break
                else:
                    next_point = node.point + next_side
                    if self.traverse_path[next_point.y][next_point.x] != 0:
                        new_node = self.traverse_path[next_point.y][next_point.x]
                    else:
                        new_node = Node(node.point + next_side,
                                        node, next_side, self)
                    if next_side == Direction.DOWN:
                        node.next_down = new_node
                    elif next_side == Direction.LEFT:
                        node.next_left = new_node
                    elif next_side == Direction.RIGHT:
                        node.next_right = new_node
                    elif next_side == Direction.UP:
                        node.next_up = new_node
                    node = new_node

            if current_side == Direction.DOWN:
                self.costs[0] = len(self.snake)/steps if steps > 0 else -1
            elif current_side == Direction.LEFT:
                self.costs[1] = len(self.snake)/steps if steps > 0 else -1
            elif current_side == Direction.RIGHT:
                self.costs[2] = len(self.snake)/steps if steps > 0 else -1
            elif current_side == Direction.UP:
                self.costs[3] = len(self.snake)/steps if steps > 0 else -1

    def towards_food(self):
        self.towards_food_direction = [
            self.is_towards_food(self.head, Direction.DOWN),
            self.is_towards_food(self.head, Direction.LEFT),
            self.is_towards_food(self.head, Direction.RIGHT),
            self.is_towards_food(self.head, Direction.UP),
        ]

    def is_towards_food(self, point: Point, direction: Direction):
        point_allowed_directions = point.allowed_directions()  # (1,1) / (-1,-1)
        if direction == Direction.DOWN and point_allowed_directions.y != 1:
            return -1
        if direction == Direction.LEFT and point_allowed_directions.x != -1:
            return -1
        if direction == Direction.RIGHT and point_allowed_directions.x != 1:
            return -1
        if direction == Direction.UP and point_allowed_directions.y != -1:
            return -1

        next_point: Point = point + direction
        if self.is_out_of_board(next_point):
            return -1
        if self.food == next_point:
            return 1

        food_direction: Point = next_point.target_directions(
            self.food)  # (0,1) / (1,0)
        next_allowed_directions = next_point.allowed_directions()  # (1,1) / (-1,-1)

        if food_direction.x == next_allowed_directions.x or food_direction.y == next_allowed_directions.y:
            return 1

        return -1

    def update_ui(self):
        self.display.fill(BLACK)

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
                # elif self.traverse_path[pt.y][pt.x] != 0:
                #     columns.append(self._get_unicode(pt))
                else:
                    columns.append('_')
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
