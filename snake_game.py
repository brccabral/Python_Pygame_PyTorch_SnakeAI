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
        if pt.x > GAME_TABLE_COLUMNS-1 or pt.x < 0 or pt.y > GAME_TABLE_ROWS-1 or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

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
                             pygame.Rect(self.head.x*BLOCK_SIZE+BLOCK_DRAW_OFFSET, self.head.y*BLOCK_SIZE+BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, BLOCK_SIZE_OFFSET))

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
