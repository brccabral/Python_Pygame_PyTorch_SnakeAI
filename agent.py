import copy
import torch
import random
import numpy as np
from collections import deque
from model import QTrainer
from snake_game import SnakeGameAI, Point
from helper import plot
from settings import MAX_MEMORY, OUTPUT_SIZE, BATCH_SIZE


class Agent:
    def __init__(self, input_size: int, hidden_size: int, lr: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate, has to be less than 1, usually 0.8-0.99
        # deque auto removes items if it gets larger than maxlen, popleft()
        self.memory_deque = deque(maxlen=MAX_MEMORY)

        self.trainer: QTrainer = QTrainer(
            lr=lr, gamma=self.gamma, input_size=input_size, hidden_size=hidden_size)

    def __repr__(self):
        return f'Agent(games:{self.number_of_games}, epsilon:{self.epsilon},' \
            f' inp:{self.input_size}, hid:{self.hidden_size}, lr:{self.lr})'

    def get_state(self, game: SnakeGameAI):
        """From the game, get some parameters and returns a list
        0-24: check if there is collision around head two steps ahead,
        4: Head row is even or odd
        5: Head column is even or odd
        6: Is food on the left
        7: Is food on the right
        8: Is food up
        9: Is food down

        Args:
            game (SnakeGameAI): the game

        Returns:
            list: booleans for each game condition
        """
        head = game.snake[0]

        # get collisions two steps ahead
        collisions = []
        for c in range(-2, 3):
            for r in range(-2, 3):
                collisions.append(game.is_collision(Point(head.x-c, head.y-r)))

        state = collisions + [

            head.x % 2,
            head.y % 2,

            # food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # memory_deque calls popleft automatically if size greater than MAX_MEMORY
        # store as a tuple containing all variables
        self.memory_deque.append(
            (state, action, reward, next_state, game_over))

    def train_long_memory(self):
        # trains in all the previous moves
        # increasing agent performance
        if len(self.memory_deque) > BATCH_SIZE:
            # list of tuples (state, action, reward, next_state, game_over)
            batch_sample = random.sample(self.memory_deque, BATCH_SIZE)
        else:
            batch_sample = self.memory_deque

        states, actions, rewards, next_states, game_overs = zip(*batch_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration vs exploitation
        self.epsilon = 80 - self.number_of_games
        action = [0 for _ in range(OUTPUT_SIZE)]
        # in the beginning this is true for some time, later self.number_of_games is larger
        # than 80 and this will never get called
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, OUTPUT_SIZE-1)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # prediction is a list of floats
            prediction = self.trainer.model.forward(state0)
            # get the larger number index
            move = torch.argmax(prediction).item()
            # set the index to 1
            action[move] = 1

        return action

    def get_play(self, state):
        self.trainer.model.eval()
        action = [0 for _ in range(OUTPUT_SIZE)]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.trainer.model.forward(state0)
        move = torch.argmax(prediction).item()
        action[move] = 1

        return action

    def copy(self):
        new_copy = Agent(self.input_size, self.hidden_size, self.lr)
        new_copy.epsilon = self.epsilon
        new_copy.number_of_games = self.number_of_games
        new_copy.memory_deque = copy.deepcopy(self.memory_deque)
        new_copy.trainer = self.trainer.copy()

        return new_copy


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
        agent.train_short_memory(
            state_old, action, reward, state_new, game_over)

        # remember
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.number_of_games += 1
            if score > best_score:
                best_score = score
                agent.trainer.model.save()

            print(
                f'Game {agent.number_of_games} Score {score} Record {best_score}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # train long memory (also called replay memory, or experience replay)
            agent.train_long_memory()


def main():
    agent = Agent()
    agent.trainer.model.load()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        action = agent.get_play(state_old)
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()


if __name__ == "__main__":
    # train()
    main()
