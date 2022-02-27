import math
import sys
import pygame

from snake_game import SnakeGameAI


pygame.init()


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
CLOCK_SPEED = 10

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('SnakeAI')
clock = pygame.time.Clock()

NUMBER_OF_AGENTS = 8
games = [SnakeGameAI() for _ in range(NUMBER_OF_AGENTS)]
square = math.ceil(math.sqrt(len(games)))

while True:

    action = [0, 0, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            action = [event.key == pygame.K_LEFT, event.key == pygame.K_UP,
                      event.key == pygame.K_RIGHT, event.key == pygame.K_DOWN]

    for index_game, game in enumerate(games):
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()

        game_x = index_game % square
        game_y = index_game // square
        game_w = SCREEN_WIDTH // square
        game_h = SCREEN_HEIGHT // square

        screen.blit(pygame.transform.scale(
            game.display, (game_w, game_h)), (game_x * game_w, game_y * game_h))

    clock.tick(CLOCK_SPEED)
