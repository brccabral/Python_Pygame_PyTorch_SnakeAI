import datetime
import sys
import argparse
from typing import List
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, MAX_GENERATIONS, WHITE, Play_Type, CLOCK_SPEED

from genetic import GeneticAlgo
from helper import plot_genetic, timer


def main(*args, **kwargs):
    pygame.init()
    if kwargs['display_gui']:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('SnakeAI')
        clock = pygame.time.Clock()

    genetic_algo = GeneticAlgo(
        number_of_agents=kwargs['number_of_agents'], play_type=kwargs['play_type'],
        lr=kwargs['lr'], input_size=kwargs['input_size'], hidden_size=kwargs['hidden_size'],
        mutation_prob=kwargs['mutation_prob'], mutation_rate=kwargs['mutation_rate'])

    user_event = None
    best_all_times: List[int] = []
    best_generation: List[int] = []

    while True:
        genetic_algo.play_step(user_event)
        # all games have ended, generation is done
        if genetic_algo.total_game_over >= kwargs['number_of_agents']:
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {genetic_algo} Generation {genetic_algo.genetic_stats.generation_count} Best All {genetic_algo.genetic_stats.best_score_all_time} Best Gen {genetic_algo.genetic_stats.best_score_generation}')

            # with timer('Plot'):
            if kwargs['plot_chart']:
                best_all_times.append(
                    genetic_algo.genetic_stats.best_score_all_time)
                best_generation.append(
                    genetic_algo.genetic_stats.best_score_generation)
                plot_genetic(best_all_times, best_generation,
                             title=genetic_algo)

            if genetic_algo.individual_save is not None:
                genetic_algo.individual_save.agent_play_type.agent.trainer.model.save(
                    file_name=f'model_{genetic_algo.individual_save.score}_{genetic_algo.genetic_stats.generation_count}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')

            if genetic_algo.has_winner():
                break

            if genetic_algo.genetic_stats.generation_count > MAX_GENERATIONS:
                print(f'Max generations reached {MAX_GENERATIONS}')
                break

            genetic_algo.reset()

        if kwargs['display_gui']:
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
                    if kwargs['play_type'] == Play_Type.USER:
                        user_event = event

            for individual in genetic_algo.population:
                if individual.order <= 15:
                    screen.blit(pygame.transform.scale(
                        individual.game.display, (individual.display_w, individual.display_h)), (individual.game_x, individual.game_y))
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--input_features', type=int, default=10, metavar='IN',
                        help='input dimension for training (default: 10)')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='H',
                        help='hidden dimension for training (default: 128)')
    parser.add_argument('--number_of_agents', type=int, default=30, metavar='AGENTS',
                        help='number of agents (default: 30)')
    parser.add_argument('--mutation_prob', type=float, default=0.01, metavar='MUTPROB',
                        help='mutation probability (default: 0.01)')
    parser.add_argument('--mutation_rate', type=float, default=0.001, metavar='MUTRATE',
                        help='mutation rate (default: 0.001)')
    parser.add_argument('--display_gui', type=bool, default=False, metavar='GUI',
                        help='display gui (default: False)')
    parser.add_argument('--plot_chart', type=bool, default=True, metavar='PLOT',
                        help='plot chart (default: True)')
    parser.add_argument('--play_type', type=int, default=3, metavar='PLAY',
                        help='play type (default: 3 - AI)')
    args = parser.parse_args()

    if args.play_type == 1:
        PLAY_TYPE = Play_Type.RANDOM
    elif args.play_type == 2:
        PLAY_TYPE = Play_Type.USER
    else:
        PLAY_TYPE = Play_Type.AI

    main(lr=args.lr, input_size=args.input_features, hidden_size=args.hidden_dim, number_of_agents=args.number_of_agents, display_gui=args.display_gui,
         plot_chart=args.plot_chart, mutation_prob=args.mutation_prob, mutation_rate=args.mutation_rate, play_type=PLAY_TYPE)
