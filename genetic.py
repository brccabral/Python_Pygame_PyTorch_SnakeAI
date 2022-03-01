import math
from typing import List
import pygame
from settings import *

from snake_game import SnakeGameAI


class Individual:
    def __init__(self, game: SnakeGameAI, order):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.score = 0
        self.number_of_individuals = NUMBER_OF_AGENTS
        self.screen_w = GAME_WIDTH
        self.screen_h = GAME_HEIGHT
        self.display_padding = GAME_DISPLAY_PADDING
        self.order = order
        self.set_display()
        self.fitness = 0

        length_x = GAME_WIDTH//BLOCK_SIZE
        length_y = GAME_HEIGHT//BLOCK_SIZE

        self.total_board_size = length_x*length_y

    def set_order(self, order):
        self.order = order
        self.set_display()

    def set_display(self):
        square = math.ceil(math.sqrt(self.number_of_individuals))
        game_column = self.order % square
        game_row = self.order // square
        self.game_w = self.screen_w // square - self.display_padding
        self.game_h = self.screen_h // square - self.display_padding
        self.game_x = game_column * \
            (self.game_w + self.display_padding) + self.display_padding
        self.game_y = game_row * (self.game_h + self.display_padding) + \
            self.display_padding

    def update_fitness(self):
        self.fitness = len(self.game.snake)/self.total_board_size

    def copy(self, order):
        return Individual(SnakeGameAI(), order=order)

    def cross_over(self, parent2, order):
        # TODO : create child from cross_over
        child = Individual(SnakeGameAI(), order)
        return child

    def mutate(self, order):
        # TODO
        return Individual(SnakeGameAI(), order)

    def __repr__(self):
        return f'Id {self.order} Score {self.score} X {self.game_x} Y {self.game_y}'


class IndividualAI(object):
    def __init__(self):
        super().__init__()


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
        text = self.font.render(
            f"Best Score All Time: {self.best_score_all_time}", True, BLACK)
        self.display.blit(text, [10, 10])
        text = self.font.render(
            f"Best Score Generation: {self.best_score_generation}", True, BLACK)
        self.display.blit(text, [10, 30])
        text = self.font.render(
            f"Best Individual: {self.best_individual}", True, BLACK)
        self.display.blit(text, [10, 50])
        text = self.font.render(
            f"Generation: {self.generation_count}", True, BLACK)
        self.display.blit(text, [10, 70])


class GeneticAlgo:
    def __init__(self, generation_limit=100):
        self.population_size = NUMBER_OF_AGENTS
        self.display_padding = GAME_DISPLAY_PADDING

        length_x = GAME_WIDTH//BLOCK_SIZE
        length_y = GAME_HEIGHT//BLOCK_SIZE

        self.total_board_size = length_x*length_y

    def generate_population(self):
        return [self.new_individual(i) for i in range(self.population_size)]

    def fitness(self, individual: Individual):
        return len(individual.game.snake)/self.total_board_size

    def new_individual(self, order):
        return Individual(SnakeGameAI(), order=order)

    def new_population(self, population: List[Individual] = None) -> List[Individual]:
        if population is None:
            return self.generate_population()

        # sort based on fitness
        population = sorted(
            population, key=lambda individual: individual.fitness, reverse=True)

        # reset order
        for order, individual in enumerate(population):
            individual.set_order(order)

        new_population: List[Individual] = []
        if len(population) == 1:
            return population
        elif len(population) == 2:
            new_population.append(population[0])
            new_population.append(self.new_individual(1))
        elif len(population) <= 4:
            new_population.append(population[0])
            new_population.append(population[1].mutate(1))
            for order in range(2, len(population)):
                new_population.append(self.new_individual(order))
        else:
            p1, p2 = population[0], population[1]
            new_population.append(p1)
            new_population.append(p1.cross_over(p2, 1))
            new_population.append(p2.cross_over(p1, 2))
            for order, individual in enumerate(population[3:len(population)//2]):
                new_population.append(individual.mutate(order + 3))
            for order in range(len(population)//2, len(population)):
                new_population.append(self.new_individual(order))

        return new_population
