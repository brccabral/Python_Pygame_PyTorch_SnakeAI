import pygame
import random
from collections import namedtuple
from settings import GAME_WIDTH, GAME_HEIGHT, GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, GREEN1, GREEN2, BLACK, BLUE1, BLUE2, BLOCK_SIZE, BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, WHITE, RED

Point = namedtuple('Point', 'x, y')


class Direction:
    RIGHT = Point(1, 0)
    LEFT = Point(-1, 0)
    UP = Point(0, -1)
    DOWN = Point(0, 1)

    @classmethod
    def get_direction(cls, current_direction: Point, action=[0, 0, 0, 0]) -> Point:
        try:
            current_index = action.index(1)
        except ValueError:
            current_index = -1

        if current_index == 0:
            return cls.LEFT
        elif current_index == 1:
            return cls.UP
        elif current_index == 2:
            return cls.RIGHT
        elif current_index == 3:
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

        self.update_cost(game)

        if self.cost < 0:
            self.next_right = -1
            self.next_left = -1
            self.next_down = -1
            self.next_up = -1

    def is_complete(self):
        return (self.next_right is not None and self.next_left is not None
                and self.next_down is not None and self.next_up is not None)

    def get_next_attempt(self, game: "SnakeGameAI"):
        food_direction = game.food_direction(self.point)

        if food_direction.x < 0:
            if self.next_left is None:
                return Direction.LEFT
        elif food_direction.x > 0:
            if self.next_right is None:
                return Direction.RIGHT

        if food_direction.y < 0:
            if self.next_up is None:
                return Direction.UP

        if self.next_down is None:
            return Direction.DOWN
        elif self.next_up is None:
            return Direction.UP
        elif self.next_right is None:
            return Direction.RIGHT
        elif self.next_left is None:
            return Direction.LEFT

        return None

    def get_next(self, game: "SnakeGameAI"):
        next_attempt = self.get_next_attempt(game)

        if next_attempt == Direction.RIGHT:
            new_node = Node(Point(self.point.x+1, self.point.y),
                            self, Direction.RIGHT, game)
            if new_node.cost < 0:
                self.next_right = -1
                return self
        elif next_attempt == Direction.LEFT:
            new_node = Node(Point(self.point.x-1, self.point.y),
                            self, Direction.LEFT, game)
            if new_node.cost < 0:
                self.next_left = -1
                return self
        elif next_attempt == Direction.DOWN:
            new_node = Node(Point(self.point.x, self.point.y+1),
                            self, Direction.DOWN, game)
            if new_node.cost < 0:
                self.next_down = -1
                return self
        elif next_attempt == Direction.UP:
            new_node = Node(Point(self.point.x, self.point.y-1),
                            self, Direction.UP, game)
            if new_node.cost < 0:
                self.next_up = -1
                return self
        else:
            new_node = self.previous
        return new_node

    def update_cost(self, game: "SnakeGameAI"):
        if self.previous is not None:
            next_cost = self.previous.cost + 1
        else:
            next_cost = 0
        self.cost = game.step_cost(self.point, next_cost)

    def __repr__(self):
        return f"Node({self.point}, cost:{self.cost}, previous:{self.previous.point})"


class SnakeGameAI:

    def __init__(self):
        # init display
        self.display = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.font = pygame.font.Font('arial.ttf', 25)

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
            action (list): list of integers that will determine where to move the snake. [left, up, right, down]

        Returns:
            tuple: reward, game_over, score
        """
        reward = 0
        if not self.game_over:
            self.count_steps += 1

            # 2. move
            self._move(action)  # update the head

            # 3. check if game over
            if self.is_collision() or self.count_steps > 2*(GAME_TABLE_ROWS*GAME_TABLE_COLUMNS):
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

    def food_direction(self, pt: Point) -> Point:
        x = self.food.x - pt.x
        if x < 0:
            x = -1
        elif x > 0:
            x = 1
        y = self.food.y - pt.y
        if y < 0:
            y = -1
        elif y > 0:
            y = 1
        return Point(x, y)

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
        point = Point(pt.x, pt.y)
        if self.is_out_of_board(point):
            return -1
        if point in self.snake:
            snake_index = self.snake.index(point)
            if snake_index + cost > len(self.snake):
                return cost
            return -1
        if point == self.food:
            cost += 1
        return cost

    def traverse_cost(self, target_point: Point, previous_point: Point, from_direction: Direction):
        step_cost = -1
        previous_node = Node(previous_point, None, None, self)
        node = Node(target_point, previous_node, from_direction, self)
        while not node.is_complete():
            node = node.get_next(self)
            if node.point == target_point and node.is_complete():
                return step_cost
            step_cost = node.cost
            if step_cost >= len(self.snake):
                break
        if step_cost < 0:
            return step_cost
        return step_cost/len(self.snake)

    def update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, GREEN1, pygame.Rect(
            self.head.x*BLOCK_SIZE, self.head.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN2,
                         pygame.Rect(self.head.x*BLOCK_SIZE+BLOCK_DRAW_OFFSET, self.head.y*BLOCK_SIZE+BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, BLOCK_SIZE_OFFSET))
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x*BLOCK_SIZE+BLOCK_DRAW_OFFSET, pt.y*BLOCK_SIZE+BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, BLOCK_SIZE_OFFSET))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x*BLOCK_SIZE, self.food.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        # pygame.display.update()

    def _move(self, action):
        """move snake head to new position

        Args:
            action (list): list of integers that will determine where to move the snake. [left, up, right, down]
        """
        self.direction = Direction.get_direction(self.direction, action)

        self.head = Point(self.head.x + self.direction.x,
                          self.head.y + self.direction.y)
        self.snake.insert(0, self.head)

    def reset(self):
        """reset is called at __init__ and by AI agent
        """

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

        self.score = 0
        self.food = None
        self._place_food()
        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.count_steps = 0

        self.game_over = False

    def get_board(self):
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
                else:
                    columns.append('_')
            rows.append("|".join(columns))
        return "\n".join(rows)
