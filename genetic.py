from abc import ABC, abstractclassmethod
import math
import torch
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

    @abstractclassmethod
    def mutate(self):
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

    def mutate(self):
        pass


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

    def mutate(self):
        pass


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

    def mutate(self):
        self_state_dict = self.agent.model.state_dict()

        linear1_rows, linear1_columns = self_state_dict['linear1.weight'].shape
        linear1_weights_probabilities = torch.tensor(
            [[random.uniform(0, 1) for _ in range(linear1_columns)] for __ in range(linear1_rows)])
        linear1_new_weights = torch.tensor(
            [[random.uniform(1-MUTATION_PROBABILITY, 1) for _ in range(linear1_columns)] for __ in range(linear1_rows)])
        linear1_new_weights = linear1_new_weights * \
            self_state_dict['linear1.weight']
        self_state_dict['linear1.weight'] = self_state_dict['linear1.weight'].where(
            linear1_weights_probabilities > MUTATION_PROBABILITY, linear1_new_weights)

        linear1_bias_probabilities = torch.tensor(
            [random.uniform(0, 1) for _ in range(linear1_rows)])
        linear1_new_bias = torch.tensor(
            [random.uniform(1-MUTATION_PROBABILITY, 1) for _ in range(linear1_rows)])
        linear1_new_bias = linear1_new_bias * self_state_dict['linear1.bias']
        self_state_dict['linear1.bias'] = self_state_dict['linear1.bias'].where(
            linear1_bias_probabilities > MUTATION_PROBABILITY, linear1_new_bias)

        linear2_rows, linear2_columns = self_state_dict['linear2.weight'].shape
        linear2_weights_probabilities = torch.tensor(
            [[random.uniform(0, 1) for _ in range(linear2_columns)] for __ in range(linear2_rows)])
        linear2_new_weights = torch.tensor(
            [[random.uniform(1-MUTATION_PROBABILITY, 1) for _ in range(linear2_columns)] for __ in range(linear2_rows)])
        linear2_new_weights = linear2_new_weights * \
            self_state_dict['linear2.weight']
        self_state_dict['linear2.weight'] = self_state_dict['linear2.weight'].where(
            linear2_weights_probabilities > MUTATION_PROBABILITY, linear2_new_weights)

        linear2_bias_probabilities = torch.tensor(
            [random.uniform(0, 1) for _ in range(linear2_rows)])
        linear2_new_bias = torch.tensor(
            [random.uniform(1-MUTATION_PROBABILITY, 1) for _ in range(linear2_rows)])
        linear2_new_bias = linear2_new_bias * self_state_dict['linear2.bias']
        self_state_dict['linear2.bias'] = self_state_dict['linear2.bias'].where(
            linear2_bias_probabilities > MUTATION_PROBABILITY, linear2_new_bias)

        self.agent.model.load_state_dict(self_state_dict)


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
        self.square = math.ceil(math.sqrt(min(self.number_of_individuals, 15)))
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
        game_column = self.order % self.square
        game_row = self.order // self.square
        self.game_w = self.screen_w // self.square - self.display_padding
        self.game_h = self.screen_h // self.square - self.display_padding
        self.game_x = game_column * \
            (self.game_w + self.display_padding) + self.display_padding
        self.game_y = game_row * (self.game_h + self.display_padding) + \
            self.display_padding

    def update_fitness(self):
        self.fitness = pow(len(self.game.snake)/self.total_board_size, 2)

    def copy(self, order):
        new_copy = Individual(SnakeGameAI(), order=order)
        new_copy.play_type = self.play_type
        return new_copy

    def cross_over(self, parent2, order):
        child = self.copy(order)
        child.play_type = self.play_type.cross_over(parent2)
        return child

    def mutate(self):
        self.play_type.mutate()

    def set_play_type(self, play_type: Agent_Play_Type):
        self.play_type = play_type

    def play_step(self, event=None):
        self.reward, self.game_over, self.score = self.play_type.play_step(
            self.game, event)

    def __repr__(self):
        return f'Order {self.order} Score {self.score} X {self.game_x} Y {self.game_y}'


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

        pop_fitness = [individual.fitness for individual in population]
        total_fitness = sum(pop_fitness)

        population[0].set_order(0)
        new_population: List[Individual] = [population[0]]
        for order in range(1, NUMBER_OF_AGENTS):
            parent1 = self.select_individual(
                population, pop_fitness, total_fitness, order)
            parent2 = self.select_individual(
                population, pop_fitness, total_fitness, order)
            child = parent1.cross_over(parent2, order)
            child.mutate()
            new_population.append(child)

        # reset order
        # for order in range(len(new_population)):
        #     new_population[order].set_order(order)

        # if len(population) == 2:
        #     new_population.append(population[0])
        #     new_population.append(self.new_individual(1))
        # elif len(population) <= 4:
        #     new_population.append(population[0])
        #     new_population.append(population[1].mutate(1))
        #     for order in range(2, len(population)):
        #         new_population.append(self.new_individual(order))
        # else:
        #     p1, p2 = population[0], population[1]
        #     new_population.append(p1)
        #     new_population.append(p1.cross_over(p2, 1))
        #     new_population.append(p2.cross_over(p1, 2))
        #     for order, individual in enumerate(population[3:len(population)//2]):
        #         new_population.append(individual.mutate(order + 3))
        #     for order in range(len(population)//2, len(population)):
        #         new_population.append(self.new_individual(order))

        return new_population

    def select_individual(self, population: List[Individual], pop_fitness: List[float], total_fitness, order):
        pick_probability = random.random()
        select = 0
        while pick_probability >= 0:
            pick_probability -= pop_fitness[select]/total_fitness
            select += 1
        select -= 1
        return population[select].copy(order)
