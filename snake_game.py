import pygame
import random
from collections import namedtuple
from settings import *


Point = namedtuple('Point', 'x, y')


class Direction:
    RIGHT = Point(1, 0)
    LEFT = Point(-1, 0)
    UP = Point(0, -1)
    DOWN = Point(0, 1)

    @classmethod
    def get_direction(cls, current_direction, action=[0, 0, 0, 0]):
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
        self.w = GAME_WIDTH
        self.h = GAME_HEIGHT
        # init display
        self.display = pygame.Surface((self.w, self.h))
        self.font = pygame.font.Font('arial.ttf', 25)

        self.reset()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE) //
                           BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE) //
                           BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """receives an action from the agent and updates the game

        Args:
            action (list): list of integers that will determine where to move the snake. [straight, right, left]

        Returns:
            tuple: reward, game_over, score
        """
        reward = 0
        if not self.game_over:
            self.frame_iteration += 1

            # 2. move
            self._move(action)  # update the head
            self.snake.insert(0, self.head)

            # 3. check if game over
            if self.is_collision() or self.frame_iteration > 100*len(self.snake):
                self.game_over = True
                reward = -10
                return reward, self.game_over, self.score

            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                self._place_food()
                reward = 10
            else:
                self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        # 6. return game over and score
        return reward, self.game_over, self.score

    def is_collision(self, pt: Point = None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, GREEN1, pygame.Rect(
            self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN2,
                         pygame.Rect(self.head.x+BLOCK_DRAW_OFFSET, self.head.y+BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, BLOCK_SIZE_OFFSET))
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(self.head.x+BLOCK_DRAW_OFFSET, self.head.y+BLOCK_DRAW_OFFSET, BLOCK_SIZE_OFFSET, BLOCK_SIZE_OFFSET))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        # pygame.display.update()

    def _move(self, action):
        """move snake head to new position

        Args:
            action (list): list of integers that will determine where to move the snake. [straight, right, left]
        """
        self.direction = Direction.get_direction(self.direction, action)

        x = self.head.x
        y = self.head.y
        x += self.direction.x*BLOCK_SIZE
        y += self.direction.y*BLOCK_SIZE

        self.head = Point(x, y)

    def reset(self):
        """reset is called at __init__ and by AI agent
        """

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.frame_iteration = 0

        self.game_over = False
