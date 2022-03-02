from abc import ABC, abstractclassmethod
import math
from os import chdir
import random
from typing import List
import pygame
from agent import Agent
from settings import *

from snake_game import SnakeGameAI


class Agent_Play_Type(ABC):
    @abstractclassmethod
    def get_action(self):
        return [0 for _ in range(OUTPUT_SIZE)]

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def play_step(self, game: SnakeGameAI, event: pygame.event.Event = None):
        pass

    @abstractclassmethod
    def cross_over(self, parent2):
        pass


class Random_Play_Type(Agent_Play_Type):
    def get_action(self):
        action = [0 for _ in range(OUTPUT_SIZE)]
        random_action_index = random.randint(0, 3)
        action[random_action_index] = 1
        return action

    def train(self):
        pass

    def play_step(self, game: SnakeGameAI, event: pygame.event.Event = None):
        action = self.get_action()
        return game.play_step(action)

    def cross_over(self, parent2):
        return self


class User_Play_Type(Agent_Play_Type):
    def get_action(self, event: pygame.event.Event = None):
        if event is not None:
            return [event.key == pygame.K_LEFT, event.key == pygame.K_UP,
                    event.key == pygame.K_RIGHT, event.key == pygame.K_DOWN]
        return [0 for _ in range(OUTPUT_SIZE)]

    def train(self):
        pass

    def play_step(self, game: SnakeGameAI, event: pygame.event.Event = None):
        action = self.get_action(event)
        return game.play_step(action)

    def cross_over(self, parent2):
        return self


class AI_Play_Type(Agent_Play_Type):
    def __init__(self, agent: Agent):
        self.agent = agent

    def get_action(self, state: List = None):
        return self.agent.get_action(state)

    def play_step(self, game: SnakeGameAI, event: pygame.event.Event = None):
        state_old = self.agent.get_state(game)
        action = self.get_action(state_old)
        reward, game_over, score = game.play_step(action)
        self.train(game, state_old, action, reward, game_over)
        return reward, game_over, score

    def train(self, game, state_old, action, reward, game_over):
        state_new = self.agent.get_state(game)
        self.agent.train_short_memory(
            state_old, action, reward, state_new, game_over)
        self.agent.remember(state_old, action, reward, state_new, game_over)

    def cross_over(self, parent2):
        self_state_dict = self.agent.model.state_dict()
        p2_state_dict = parent2.play_type.agent.model.state_dict()

        cross1 = random.randint(
            1, self_state_dict['linear1.weight'].shape[1] - 1)
        cross2 = random.randint(
            1, self_state_dict['linear2.weight'].shape[1] - 1)

        child_state_dict = self_state_dict
        child_state_dict['linear1.weight'][:,
                                           cross1:] = p2_state_dict['linear1.weight'][:, cross1:]
        child_state_dict['linear1.bias'][cross1:] = p2_state_dict['linear1.bias'][cross1:]
        child_state_dict['linear2.weight'][:,
                                           cross2:] = p2_state_dict['linear2.weight'][:, cross2:]
        child_state_dict['linear2.bias'][cross2:] = p2_state_dict['linear2.bias'][cross2:]

        child = AI_Play_Type(Agent())
        child.agent.model.load_state_dict(child_state_dict)
        child.agent.epsilon = self.agent.epsilon
        child.agent.number_of_games = self.agent.number_of_games
        child.agent.memory_deque = self.agent.memory_deque
        
        return child


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

        if PLAY_TYPE == Play_Type.RANDOM:
            self.set_play_type(Random_Play_Type())
        elif PLAY_TYPE == Play_Type.AI:
            self.set_play_type(AI_Play_Type(Agent()))
        elif PLAY_TYPE == Play_Type.USER:
            self.set_play_type(User_Play_Type())

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
        # print(f'cross {order}')
        child = Individual(SnakeGameAI(), order)
        child.play_type = self.play_type.cross_over(parent2)
        return child

    def mutate(self, order):
        # TODO
        return Individual(SnakeGameAI(), order)

    def set_play_type(self, play_type: Agent_Play_Type):
        self.play_type = play_type

    def play_step(self, event=None):
        self.reward, self.game_over, self.score = self.play_type.play_step(
            self.game, event)

    def __repr__(self):
        return f'Id {self.order} Score {self.score} X {self.game_x} Y {self.game_y}'


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
        if len(population) == 1:
            return population

        # sort based on fitness
        population = sorted(
            population, key=lambda individual: individual.fitness, reverse=True)

        # reset order
        for order, individual in enumerate(population):
            individual.set_order(order)

        new_population: List[Individual] = []
        if len(population) == 2:
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
