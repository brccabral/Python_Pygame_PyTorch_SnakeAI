import sys
import pygame
from settings import *

from genetic import GeneticAlgo, GeneticStats, Individual

pygame.init()


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('SnakeAI')
clock = pygame.time.Clock()

genetic_algo = GeneticAlgo()

population = genetic_algo.new_population()
genetic_stats = GeneticStats(NUMBER_OF_AGENTS)
best_individual_generation: Individual = population[0]


while True:
    screen.fill(WHITE)
    user_event = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if PLAY_TYPE == Play_Type.USER:
                user_event = event

    total_game_over = 0
    for individual in population:

        individual.play_step(user_event)
        individual.update_fitness()

        if individual.game_over:
            total_game_over += 1

        if individual.score > genetic_stats.best_score_all_time:
            genetic_stats.best_score_all_time = individual.score

        if individual.score > genetic_stats.best_score_generation:
            best_individual_generation = individual
            genetic_stats.best_score_generation = best_individual_generation.score
            genetic_stats.best_individual = best_individual_generation.order

        if individual.order <= 15:
            individual.game.update_ui()
            screen.blit(pygame.transform.scale(
                individual.game.display, (individual.game_w, individual.game_h)), (individual.game_x, individual.game_y))

        genetic_stats.update_ui()
        screen.blit(pygame.transform.scale(genetic_stats.display,
                    (genetic_stats.w, genetic_stats.h)), (SCREEN_WIDTH - 450, 0))

        screen.blit(pygame.transform.scale(best_individual_generation.game.display,
                    (400, 240)), (SCREEN_WIDTH - 450, 200))

    # all games have ended, generation is done
    if total_game_over == len(population):
        population = sorted(
            population, key=lambda individual: individual.fitness, reverse=True)
        if population[0].fitness >= 1:
            winner = population[0]
            print(
                f'Generation {genetic_stats.generation_count} has a winner ID {population[0].order}')
            break

        if genetic_stats.generation_count > MAX_GENERATIONS:
            print(f'Max generations reached {MAX_GENERATIONS}')
            break

        genetic_stats.best_score_generation = 0
        genetic_stats.generation_count += 1
        for individual in population:
            individual.game.reset()
            individual.play_type.agent.number_of_games += 1

        population = genetic_algo.new_population(population)
        best_individual_generation = population[0]
        genetic_stats.best_individual = best_individual_generation.order

    pygame.display.update()
    clock.tick(CLOCK_SPEED)

# TODO : save winner

print(
    f'best_score_all_time {genetic_stats.best_score_all_time} generation_count {genetic_stats.generation_count}')

pygame.quit()
