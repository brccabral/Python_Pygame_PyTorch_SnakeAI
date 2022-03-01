import sys
import pygame
from settings import *

from genetic import AI_Play_Type, GeneticAlgo, GeneticStats, Individual, Random_Play_Type, User_Play_Type

pygame.init()


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('SnakeAI')
clock = pygame.time.Clock()

genetic_algo = GeneticAlgo()

population = genetic_algo.new_population()
genetic_stats = GeneticStats(NUMBER_OF_AGENTS)
best_individual_generation: Individual = population[0]


agent_play_type = User_Play_Type()
# agent_play_type = Agent_Play_Type()


while True:
    screen.fill(WHITE)
    action = [0 for _ in range(OUTPUT_SIZE)]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if PLAY_TYPE == Play_Type.USER:
                action = agent_play_type.get_action(event)

    total_game_over = 0
    for individual in population:
        if PLAY_TYPE != Play_Type.USER:
            action = individual.agent.get_action()

        individual.reward, individual.game_over, individual.score = individual.game.play_step(
            action)
        individual.update_fitness()

        if individual.game_over:
            total_game_over += 1

        if individual.score > genetic_stats.best_score_all_time:
            genetic_stats.best_score_all_time = individual.score

        if individual.score > genetic_stats.best_score_generation:
            genetic_stats.best_score_generation = individual.score
            genetic_stats.best_individual = individual.order
            best_individual_generation = individual

        if individual.order <= 15:
            individual.game.update_ui()
            screen.blit(pygame.transform.scale(
                individual.game.display, (individual.game_w, individual.game_h)), (individual.game_x, individual.game_y))

        genetic_stats.update_ui()
        screen.blit(pygame.transform.scale(genetic_stats.display,
                    (genetic_stats.w, genetic_stats.h)), (SCREEN_WIDTH - 450, 0))

        screen.blit(pygame.transform.scale(best_individual_generation.game.display,
                    (individual.game_w, individual.game_h)), (SCREEN_WIDTH - 450, 200))

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
            break

        genetic_stats.best_score_generation = 0
        genetic_stats.generation_count += 1
        for individual in population:
            individual.game.reset()

        population = genetic_algo.new_population(population)

    pygame.display.update()
    clock.tick(CLOCK_SPEED)

# TODO : save winner

pygame.quit()
