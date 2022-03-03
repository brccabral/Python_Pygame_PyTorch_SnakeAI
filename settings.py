# rgb colors
from enum import Enum

# game settings
BLOCK_SIZE = 20
BLOCK_DRAW_OFFSET = int(BLOCK_SIZE*0.2)
BLOCK_SIZE_OFFSET = int(BLOCK_SIZE*0.6)
GAME_TABLE_COLUMNS = 40
GAME_TABLE_ROWS = 24

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0, 0, 0)

# game display
GAME_WIDTH = GAME_TABLE_COLUMNS * BLOCK_SIZE
GAME_HEIGHT = GAME_TABLE_ROWS * BLOCK_SIZE
GAME_DISPLAY_PADDING = 2

# main display
DISPLAY_GUI = False
PLOT_CHART = True
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 480
CLOCK_SPEED = 120

# genetic
MAX_GENERATIONS = 100_000
NUMBER_OF_AGENTS = 30
MUTATION_PROBABILITY = 0.01
MUTATION_RATE = 0.001
FITNESS_TARGET = pow(GAME_TABLE_COLUMNS * GAME_TABLE_ROWS, 2)


class Play_Type(Enum):
    RANDOM = 1
    USER = 2
    AI = 3


PLAY_TYPE = Play_Type.AI

# agent training
MAX_MEMORY = 100_000  # how many previous moves will be stored in memory_deque
BATCH_SIZE = 1000
LR = 0.001
INPUT_SIZE = 30  # has to be the length of Agent.get_state
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4  # has to be the number of possible actions, Agent.get_action
