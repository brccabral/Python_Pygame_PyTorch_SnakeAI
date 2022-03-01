# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0, 0, 0)

# game settings
BLOCK_SIZE = 20
BLOCK_DRAW_OFFSET = int(BLOCK_SIZE*0.2)
BLOCK_SIZE_OFFSET = int(BLOCK_SIZE*0.6)

# game display
GAME_WIDTH = 800
GAME_HEIGHT = 480
GAME_DISPLAY_PADDING = 2

# main display
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 480
CLOCK_SPEED = 10

# genetic
MAX_GENERATIONS = 100
NUMBER_OF_AGENTS = 16

# agent training
MAX_MEMORY = 100_000 # how many previous moves will be stored in memory_deque
BATCH_SIZE = 1000
LR = 0.001
INPUT_SIZE = 11  # has to be the length of Agent.get_state
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3  # has to be the number of possible actions, Agent.get_action
