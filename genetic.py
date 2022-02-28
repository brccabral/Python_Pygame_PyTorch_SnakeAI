import math
import pygame

from snake_game import BLACK, WHITE, SnakeGameAI


class Individual:
    def __init__(self, game: SnakeGameAI, id, number_of_individuals, screen_w, screen_h, display_padding):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.score = 0
        self.id = id
        self.number_of_individuals = number_of_individuals
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.display_padding = display_padding
        self.set_display()

    def set_display(self):
        square = math.ceil(math.sqrt(self.number_of_individuals))
        game_column = self.id % square
        game_row = self.id // square
        self.game_w = self.screen_w // square - self.display_padding
        self.game_h = self.screen_h // square - self.display_padding
        self.game_x = game_column * \
            (self.game_w + self.display_padding) + self.display_padding
        self.game_y = game_row * (self.game_h + self.display_padding) + \
            self.display_padding

    def __repr__(self):
        return f'Id {self.id} Score {self.score} X {self.game_x} Y {self.game_y}'


class GeneticStats:
    def __init__(self, population_size=1, w=350, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.Surface((self.w, self.h))
        self.font = pygame.font.Font('arial.ttf', 25)
        self.population_size = population_size
        self.best_score_all_time = 0
        self.best_score_generation = 0
        self.best_individual = 0
        self.generation_count = 0

    def update_ui(self):
        self.display.fill(WHITE)
        text = self.font.render(f"Best Score All Time: {self.best_score_all_time}", True, BLACK)
        self.display.blit(text, [10, 10])
        text = self.font.render(f"Best Score Generation: {self.best_score_generation}", True, BLACK)
        self.display.blit(text, [10, 30])
        text = self.font.render(f"Best Individual: {self.best_individual}", True, BLACK)
        self.display.blit(text, [10, 50])
        text = self.font.render(f"Generation: {self.generation_count}", True, BLACK)
        self.display.blit(text, [10, 70])
