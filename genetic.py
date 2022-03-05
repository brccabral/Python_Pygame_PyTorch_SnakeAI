from abc import ABC, abstractclassmethod
import datetime
import math
import torch
import random
from typing import List
import pygame
from agent import Agent
from settings import Play_Type, BLOCK_SIZE, GAME_WIDTH, GAME_HEIGHT, GAME_DISPLAY_PADDING, OUTPUT_SIZE, WHITE, BLACK, FITNESS_TARGET

from snake_game import SnakeGameAI


class Agent_Play_Type(ABC):
    def __init__(self):
        self.agent: Agent = None

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

    @abstractclassmethod
    def copy(self):
        pass


class Random_Play_Type(Agent_Play_Type):
    def __init__(self):
        self.agent: Agent = None

    def get_action(self):
        action = [0 for _ in range(OUTPUT_SIZE)]
        random_action_index = random.randint(0, OUTPUT_SIZE-1)
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

    def copy(self):
        pass


class User_Play_Type(Agent_Play_Type):
    def __init__(self):
        self.agent: Agent = None

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

    def copy(self):
        pass


class AI_Play_Type(Agent_Play_Type):
    def __init__(self, agent: Agent, lr: float, input_size: int, hidden_size: int, mutation_probability: float, mutation_rate: float):
        self.agent = agent
        self.mutation_probability = mutation_probability
        self.mutation_rate = mutation_rate
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __repr__(self):
        return f'AI({self.agent}, mut_prob:{self.mutation_probability}, mut_rate:{self.mutation_rate})'

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
        self_state_dict = self.agent.trainer.model.state_dict()
        p2_state_dict = parent2.agent_play_type.agent.trainer.model.state_dict()

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

        child = AI_Play_Type(Agent(input_size=self.input_size, hidden_size=self.hidden_size,
                                   lr=self.lr), lr=self.lr, input_size=self.input_size, hidden_size=self.hidden_size,
                             mutation_probability=self.mutation_probability, mutation_rate=self.mutation_rate)
        child.agent.trainer.model.load_state_dict(child_state_dict)
        child.agent.epsilon = self.agent.epsilon
        child.agent.number_of_games = self.agent.number_of_games
        child.agent.memory_deque = self.agent.memory_deque.copy()

        return child

    def mutate(self):
        self_state_dict = self.agent.trainer.model.state_dict()

        linear1_rows, linear1_columns = self_state_dict['linear1.weight'].shape
        linear1_weights_probabilities = torch.tensor(
            [[random.uniform(0, 1) for _ in range(linear1_columns)] for __ in range(linear1_rows)])
        linear1_new_weights = torch.tensor(
            [[random.uniform(1-self.mutation_rate, 1) for _ in range(linear1_columns)] for __ in range(linear1_rows)])
        linear1_new_weights = linear1_new_weights * \
            self_state_dict['linear1.weight']
        self_state_dict['linear1.weight'] = self_state_dict['linear1.weight'].where(
            linear1_weights_probabilities > self.mutation_probability, linear1_new_weights)

        linear1_bias_probabilities = torch.tensor(
            [random.uniform(0, 1) for _ in range(linear1_rows)])
        linear1_new_bias = torch.tensor(
            [random.uniform(1-self.mutation_rate, 1) for _ in range(linear1_rows)])
        linear1_new_bias = linear1_new_bias * self_state_dict['linear1.bias']
        self_state_dict['linear1.bias'] = self_state_dict['linear1.bias'].where(
            linear1_bias_probabilities > self.mutation_probability, linear1_new_bias)

        linear2_rows, linear2_columns = self_state_dict['linear2.weight'].shape
        linear2_weights_probabilities = torch.tensor(
            [[random.uniform(0, 1) for _ in range(linear2_columns)] for __ in range(linear2_rows)])
        linear2_new_weights = torch.tensor(
            [[random.uniform(1-self.mutation_rate, 1) for _ in range(linear2_columns)] for __ in range(linear2_rows)])
        linear2_new_weights = linear2_new_weights * \
            self_state_dict['linear2.weight']
        self_state_dict['linear2.weight'] = self_state_dict['linear2.weight'].where(
            linear2_weights_probabilities > self.mutation_probability, linear2_new_weights)

        linear2_bias_probabilities = torch.tensor(
            [random.uniform(0, 1) for _ in range(linear2_rows)])
        linear2_new_bias = torch.tensor(
            [random.uniform(1-self.mutation_rate, 1) for _ in range(linear2_rows)])
        linear2_new_bias = linear2_new_bias * self_state_dict['linear2.bias']
        self_state_dict['linear2.bias'] = self_state_dict['linear2.bias'].where(
            linear2_bias_probabilities > self.mutation_probability, linear2_new_bias)

        self.agent.trainer.model.load_state_dict(self_state_dict)

    def copy(self):
        new_copy = AI_Play_Type(Agent(self.input_size, self.hidden_size, self.lr), lr=self.lr,
                                input_size=self.input_size, hidden_size=self.hidden_size,
                                mutation_probability=self.mutation_probability, mutation_rate=self.mutation_rate)
        new_copy.agent = self.agent.copy()

        return new_copy


class Individual:
    def __init__(self, game: SnakeGameAI, order: int, number_of_agents: int, play_type: Play_Type,
                 lr: float, mutation_prob: float, mutation_rate: float, input_size: int, hidden_size: int):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.score = 0
        self.number_of_agents = number_of_agents
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.play_type = play_type
        self.game_width = GAME_WIDTH
        self.game_height = GAME_HEIGHT
        self.display_padding = GAME_DISPLAY_PADDING
        self.square = math.ceil(math.sqrt(min(self.number_of_agents, 15)))
        self.display_w = self.game_width // self.square - self.display_padding
        self.display_h = self.game_height // self.square - self.display_padding
        self.order = order
        self.set_display()
        self.fitness = 0

        length_x = self.game_width//BLOCK_SIZE
        length_y = self.game_height//BLOCK_SIZE

        self.total_board_size = length_x*length_y

        if play_type == Play_Type.RANDOM:
            self.set_agent_play_type(Random_Play_Type())
        elif play_type == Play_Type.USER:
            self.set_agent_play_type(User_Play_Type())
        else:
            self.set_agent_play_type(AI_Play_Type(Agent(input_size=self.input_size, hidden_size=self.hidden_size,
                                     lr=self.lr), lr=self.lr, input_size=self.input_size, hidden_size=self.hidden_size,
                                     mutation_probability=self.mutation_prob, mutation_rate=self.mutation_rate))

    def set_order(self, order):
        self.order = order
        self.set_display()

    def set_display(self):
        game_column = self.order % self.square
        game_row = self.order // self.square

        self.game_x = game_column * \
            (self.display_w + self.display_padding) + self.display_padding
        self.game_y = game_row * (self.display_h + self.display_padding) + \
            self.display_padding

    def update_fitness(self):
        self.fitness = pow(len(self.game.snake), 2)

    def copy(self, order):
        new_copy = Individual(SnakeGameAI(), order=order, number_of_agents=self.number_of_agents,
                              play_type=self.play_type, lr=self.lr, mutation_prob=self.mutation_prob,
                              mutation_rate=self.mutation_rate, input_size=self.input_size, hidden_size=self.hidden_size)
        new_copy.agent_play_type = self.agent_play_type.copy()

        return new_copy

    def cross_over(self, parent2, order):
        child = self.copy(order)
        child.agent_play_type = self.agent_play_type.cross_over(parent2)
        return child

    def mutate(self):
        self.agent_play_type.mutate()

    def set_agent_play_type(self, agent_play_type: Agent_Play_Type):
        self.agent_play_type = agent_play_type

    def play_step(self, event=None):
        self.reward, self.game_over, self.score = self.agent_play_type.play_step(
            self.game, event)
        self.update_fitness()

    def reset(self):
        self.score = 0
        self.game.reset()
        self.agent_play_type.agent.number_of_games += 1
        self.agent_play_type.agent.train_long_memory()
        self.game_over = False

    def update_ui(self):
        self.game.update_ui()

    def __repr__(self):
        return f'Order {self.order} Score {self.score} {self.agent_play_type}'


class GeneticStats:
    def __init__(self, w: int = 350, h: int = 480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.Surface((self.w, self.h))
        self.font = pygame.font.Font('arial.ttf', 25)
        self.best_score_all_time = 0
        self.best_score_generation = 0
        self.best_individual_order: int = 0
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
            f"Best Individual: {self.best_individual_order}", True, BLACK)
        self.display.blit(text, [10, 50])
        text = self.font.render(
            f"Generation: {self.generation_count}", True, BLACK)
        self.display.blit(text, [10, 70])


class GeneticAlgo:
    def __init__(self, number_of_agents: int, play_type: Play_Type, lr: float, mutation_prob: float, mutation_rate: float, input_size: int, hidden_size: int):
        self.population_size = number_of_agents
        self.play_type = play_type
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate

        length_x = GAME_WIDTH//BLOCK_SIZE
        length_y = GAME_HEIGHT//BLOCK_SIZE

        self.total_board_size = length_x*length_y

        self.generate_population()
        self.individual_highlight: Individual = self.population[0]
        self.individual_save: Individual = None
        self.total_game_over = 0

        self.genetic_stats = GeneticStats()

    def __repr__(self):
        return f'Genetic Pop={self.population_size} LR={self.lr} Mut_Prob={self.mutation_prob}, Mut_Rate={self.mutation_rate}'

    def generate_population(self):
        self.population = [self.new_individual(
            i) for i in range(self.population_size)]

    def new_individual(self, order):
        return Individual(SnakeGameAI(), order=order, number_of_agents=self.population_size, play_type=self.play_type, lr=self.lr, mutation_prob=self.mutation_prob, mutation_rate=self.mutation_rate, input_size=self.input_size, hidden_size=self.hidden_size)

    def new_population(self) -> List[Individual]:
        if len(self.population) == 1:
            return

        # self.evolution(population)
        self.natural_selection()
        self.individual_highlight = self.population[0]
        self.genetic_stats.best_individual_order = 0

    def natural_selection(self):
        pop_fitness = [individual.fitness for individual in self.population]
        total_fitness = sum(pop_fitness)

        new_population: List[Individual] = []

        new_population.append(self.population[0].copy(0))

        for order in range(1, self.population_size):
            selected = self.select_individual(pop_fitness, total_fitness)
            new_population.append(self.population[selected].copy(order))

        self.population = new_population

    def evolution(self):
        pop_fitness = [individual.fitness for individual in self.population]
        total_fitness = sum(pop_fitness)

        self.population[0].set_order(0)

        for order in range(1, self.population_size):
            selected1 = self.select_individual(
                pop_fitness, total_fitness)
            selected2 = self.select_individual(
                pop_fitness, total_fitness)
            child = self.population[selected1].cross_over(
                self.population[selected2], order)
            child.mutate()
            child.set_order(order)
            self.population[order] = child

    def select_individual(self, pop_fitness: List[float], total_fitness):
        pick_probability = random.random()
        select = 0
        while pick_probability >= 0:
            pick_probability -= pop_fitness[select]/total_fitness
            select += 1
        select -= 1
        return select

    def play_step(self, user_event: List[int]):

        for individual in self.population:

            if individual.game_over:
                self.total_game_over += 1
                continue

            if self.individual_highlight.game_over:
                self.individual_highlight = individual
            individual.play_step(user_event)

            if individual.score > self.genetic_stats.best_score_all_time:
                self.genetic_stats.best_score_all_time = individual.score
                self.individual_save = individual

            if individual.score > self.genetic_stats.best_score_generation:
                self.individual_highlight = individual
                self.genetic_stats.best_score_generation = individual.score
                self.genetic_stats.best_individual_order = individual.order

    def has_winner(self):
        self.population = sorted(
            self.population, key=lambda individual: individual.fitness, reverse=True)
        if self.population[0].fitness >= FITNESS_TARGET:
            self.population[0].agent_play_type.agent.trainer.model.save(
                file_name=f'model_winner_{self.genetic_stats.generation_count}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
            print(
                f'Generation {self.genetic_stats.generation_count} has a winner ID {self.population[0].order}')

    def reset(self):
        self.total_game_over = 0
        self.individual_save = None
        for individual in self.population:
            individual.reset()

        self.genetic_stats.best_score_generation = 0
        self.genetic_stats.generation_count += 1

        self.new_population()

    def update_ui(self):
        self.genetic_stats.update_ui()

        for individual in self.population:
            if individual.order <= 15:
                individual.update_ui()
            else:
                break
        self.individual_highlight.update_ui()
