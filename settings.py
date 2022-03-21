# rgb colors
from enum import Enum

# game settings
BLOCK_SIZE = 20
BLOCK_DRAW_OFFSET = int(BLOCK_SIZE*0.2)
BLOCK_SIZE_OFFSET = int(BLOCK_SIZE*0.6)
GAME_TABLE_COLUMNS = 20
GAME_TABLE_ROWS = 20

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLACK = (0, 0, 0)

# game display
GAME_WIDTH = GAME_TABLE_COLUMNS * BLOCK_SIZE
GAME_HEIGHT = GAME_TABLE_ROWS * BLOCK_SIZE
GAME_DISPLAY_PADDING = 2

# main display
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 550
CLOCK_SPEED = 120

# genetic
MAX_GENERATIONS = 100_000
FITNESS_TARGET = pow(GAME_TABLE_COLUMNS * GAME_TABLE_ROWS, 2)


class Play_Type(Enum):
    RANDOM = 1
    USER = 2
    AI = 3


# agent training
MAX_MEMORY = 100_000  # how many previous moves will be stored in memory_deque
BATCH_SIZE = 1_200
OUTPUT_SIZE = 4  # has to be the number of possible actions, Agent.get_action
