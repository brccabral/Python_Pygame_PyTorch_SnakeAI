import math
from typing import List
import pygame

from snake_game import SnakeGameAI


class Individual:
    def __init__(self, game: SnakeGameAI, id, number_of_individuals, display_padding):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.score = 0
        self.id = id
        self.number_of_individuals = number_of_individuals
        self.screen_w = SnakeGameAI.GAME_WIDTH
        self.screen_h = SnakeGameAI.GAME_HEIGHT
        self.display_padding = display_padding
        self.set_display()
        self.fitness = 0

        length_x = SnakeGameAI.GAME_WIDTH//SnakeGameAI.BLOCK_SIZE
        length_y = SnakeGameAI.GAME_HEIGHT//SnakeGameAI.BLOCK_SIZE

        self.total_board_size = length_x*length_y

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

    def update_fitness(self):
        self.fitness = len(self.game.snake)/self.total_board_size

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
        self.display.fill(SnakeGameAI.WHITE)
        text = self.font.render(
            f"Best Score All Time: {self.best_score_all_time}", True, SnakeGameAI.BLACK)
        self.display.blit(text, [10, 10])
        text = self.font.render(
            f"Best Score Generation: {self.best_score_generation}", True, SnakeGameAI.BLACK)
        self.display.blit(text, [10, 30])
        text = self.font.render(
            f"Best Individual: {self.best_individual}", True, SnakeGameAI.BLACK)
        self.display.blit(text, [10, 50])
        text = self.font.render(
            f"Generation: {self.generation_count}", True, SnakeGameAI.BLACK)
        self.display.blit(text, [10, 70])


class GeneticAlgo:
    def __init__(self, population_size=1, display_padding=2, generation_limit=100):
        self.population_size = population_size
        self.display_padding = display_padding

        length_x = SnakeGameAI.GAME_WIDTH//SnakeGameAI.BLOCK_SIZE
        length_y = SnakeGameAI.GAME_HEIGHT//SnakeGameAI.BLOCK_SIZE

        self.total_board_size = length_x*length_y

    def generate_population(self):
        return [self.new_individual(i) for i in range(self.population_size)]

    def fitness(self, individual: Individual):
        return len(individual.game.snake)/self.total_board_size

    def new_individual(self, id):
        return Individual(SnakeGameAI(), id=id, number_of_individuals=self.population_size, display_padding=self.display_padding)

