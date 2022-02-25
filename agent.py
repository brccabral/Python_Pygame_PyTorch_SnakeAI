import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

# how many previous moves will be stored in memory_deque
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamme = 0 # discount rate
        # deque auto removes items if it gets larger than maxlen, popleft()
        self.memory_deque = deque(maxlen=MAX_MEMORY)
        # TODO: model, trainer

    def get_state(self, game: SnakeGameAI):
        pass

    def remember(self, state, action, reward, next_state, game_over):
        pass

    def train_long_memory(self):
        # trains in all the previous moves
        # increasing agent performance
        pass

    def train_short_memory(self, state, action, reward, next_state, game_over):
        pass

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)

        # perform action, play game, and get new state
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        # remember
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.number_of_games += 1
            if score > best_score:
                best_score = score
            print(f'Game {agent.number_of_games} Score {score} Record {best_score}')

            # train long memory (also called replay memory, or experience replay)
            agent.train_long_memory()




if __name__ == "__main__":
    train()
