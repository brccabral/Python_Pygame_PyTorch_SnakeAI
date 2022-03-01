import math
import random
import sys
from typing import List
import pygame

from snake_game import BLACK, WHITE, SnakeGameAI
from agent import Agent
from genetic import GeneticStats, Individual

pygame.init()


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 480
GAME_WIDTH = 800
GAME_HEIGHT = 480
CLOCK_SPEED = 10
GAME_DISPLAY_PADDING = 2

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('SnakeAI')
clock = pygame.time.Clock()

NUMBER_OF_AGENTS = 2
population = [Individual(SnakeGameAI(), id=i, number_of_individuals=NUMBER_OF_AGENTS, screen_w=GAME_WIDTH,
                         screen_h=GAME_HEIGHT, display_padding=GAME_DISPLAY_PADDING) for i in range(NUMBER_OF_AGENTS)]
genetic_stats = GeneticStats(NUMBER_OF_AGENTS)
best_individual_generation: Individual = population[0]


class Play_Type():
    def get_action(self, state: List = None):
        action = [0, 0, 0, 0]
        random_action_index = random.randint(0, 3)
        action[random_action_index] = 1
        return action


class User_Play_Type(Play_Type):
    def get_action(self, event: pygame.event.Event = None):
        return [event.key == pygame.K_LEFT, event.key == pygame.K_UP,
                event.key == pygame.K_RIGHT, event.key == pygame.K_DOWN]


class Agent_Play_Type(Play_Type):
    def __init__(self, agent: Agent):
        self.agent = agent

    def get_action(self, state: List = None):
        return self.agent.get_action(state)


play_type = User_Play_Type()
# play_type = Play_Type()

while True:
    screen.fill(WHITE)
    action = [0, 0, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if type(play_type) == User_Play_Type:
                action = play_type.get_action(event)

    total_game_over = 0
    for index_game, individual in enumerate(population):
        if type(play_type) == Play_Type:
            action = play_type.get_action()

        individual.reward, individual.game_over, individual.score = individual.game.play_step(
            action)

        if individual.game_over:
            total_game_over += 1

        if individual.score > genetic_stats.best_score_all_time:
            genetic_stats.best_score_all_time = individual.score

        if individual.score > genetic_stats.best_score_generation:
            genetic_stats.best_score_generation = individual.score
            genetic_stats.best_individual = individual.id
            best_individual_generation = individual

        screen.blit(pygame.transform.scale(
            individual.game.display, (individual.game_w, individual.game_h)), (individual.game_x, individual.game_y))

        genetic_stats.update_ui()
        screen.blit(pygame.transform.scale(genetic_stats.display,
                    (genetic_stats.w, genetic_stats.h)), (SCREEN_WIDTH - 450, 0))

        screen.blit(pygame.transform.scale(best_individual_generation.game.display,
                    (individual.game_w, individual.game_h)), (SCREEN_WIDTH - 450, 200))

    if total_game_over == len(population):
        genetic_stats.best_score_generation = 0
        genetic_stats.generation_count += 1
        for individual in population:
            individual.game.reset()

    pygame.display.update()
    clock.tick(CLOCK_SPEED)
