import datetime
import sys
from typing import List
import pygame
from settings import *

from genetic import GeneticAlgo
from helper import plot_genetic, timer

pygame.init()
if DISPLAY_GUI:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('SnakeAI')
    clock = pygame.time.Clock()

genetic_algo = GeneticAlgo()

user_event = None
best_all_times: List[int] = []
best_generation: List[int] = []

while True:
    genetic_algo.play_step(user_event)
    # all games have ended, generation is done
    if genetic_algo.total_game_over >= NUMBER_OF_AGENTS:
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Generation {genetic_algo.genetic_stats.generation_count} Best All {genetic_algo.genetic_stats.best_score_all_time} Best Gen {genetic_algo.genetic_stats.best_score_generation}')

        # with timer('Plot'):
        if PLOT_CHART:
            best_all_times.append(
                genetic_algo.genetic_stats.best_score_all_time)
            best_generation.append(
                genetic_algo.genetic_stats.best_score_generation)
            plot_genetic(best_all_times, best_generation)

        if genetic_algo.individual_save is not None:
            genetic_algo.individual_save.play_type.agent.model.save(
                file_name=f'model_{genetic_algo.individual_save.score}_{genetic_algo.genetic_stats.generation_count}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')

        if genetic_algo.has_winner():
            break

        if genetic_algo.genetic_stats.generation_count > MAX_GENERATIONS:
            print(f'Max generations reached {MAX_GENERATIONS}')
            break

        genetic_algo.reset()

    if DISPLAY_GUI:
        genetic_algo.update_ui()
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

        for individual in genetic_algo.population:
            if individual.order <= 15:
                screen.blit(pygame.transform.scale(
                    individual.game.display, (individual.game_w, individual.game_h)), (individual.game_x, individual.game_y))
            else:
                break

        screen.blit(pygame.transform.scale(genetic_algo.genetic_stats.display,
                    (genetic_algo.genetic_stats.w, genetic_algo.genetic_stats.h)), (SCREEN_WIDTH - 450, 0))

        screen.blit(pygame.transform.scale(genetic_algo.individual_highlight.game.display,
                    (400, 240)), (SCREEN_WIDTH - 450, 200))

        pygame.display.update()
        clock.tick(CLOCK_SPEED)


print(
    f'best_score_all_time {genetic_algo.genetic_stats.best_score_all_time} generation_count {genetic_algo.genetic_stats.generation_count}')

pygame.quit()
