import datetime
import sys
import pygame
from settings import *

from genetic import GeneticAlgo, GeneticStats, Individual

pygame.init()
if DISPLAY_GUI:

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('SnakeAI')
    clock = pygame.time.Clock()

genetic_algo = GeneticAlgo()

population = genetic_algo.new_population()
genetic_stats = GeneticStats(NUMBER_OF_AGENTS)
individual_highlight: Individual = population[0]
individual_save: Individual = None

user_event = None

while True:
    if DISPLAY_GUI:
        user_event = None
        screen.fill(WHITE)
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
        elif individual_highlight.game_over:
            individual_highlight = individual

        if individual.score > genetic_stats.best_score_all_time:
            genetic_stats.best_score_all_time = individual.score
            individual_save = individual

        if individual.score > genetic_stats.best_score_generation:
            individual_highlight = individual
            genetic_stats.best_score_generation = individual.score
            genetic_stats.best_individual = individual.order

        if DISPLAY_GUI:
            if individual.order <= 15:
                individual.game.update_ui()
                screen.blit(pygame.transform.scale(
                    individual.game.display, (individual.game_w, individual.game_h)), (individual.game_x, individual.game_y))

            genetic_stats.update_ui()
            screen.blit(pygame.transform.scale(genetic_stats.display,
                        (genetic_stats.w, genetic_stats.h)), (SCREEN_WIDTH - 450, 0))

            individual_highlight.game.update_ui()
            screen.blit(pygame.transform.scale(individual_highlight.game.display,
                        (400, 240)), (SCREEN_WIDTH - 450, 200))

    # all games have ended, generation is done
    if total_game_over == len(population):
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Generation {genetic_stats.generation_count} Best All {genetic_stats.best_score_all_time} Best Gen {genetic_stats.best_score_generation}')

        if individual_save is not None:
            individual_save.play_type.agent.model.save(
                file_name=f'model_{individual_save.score}_{genetic_stats.generation_count}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
            individual_save = None

        population = sorted(
            population, key=lambda individual: individual.fitness, reverse=True)
        if population[0].fitness >= pow(population[0].total_board_size, 2):
            winner = population[0]
            winner.play_type.agent.model.save(
                file_name=f'model_winner_{genetic_stats.generation_count}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
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
        individual_highlight = population[0]
        genetic_stats.best_individual = individual_highlight.order

    if DISPLAY_GUI:
        pygame.display.update()
        clock.tick(CLOCK_SPEED)


print(
    f'best_score_all_time {genetic_stats.best_score_all_time} generation_count {genetic_stats.generation_count}')

pygame.quit()
