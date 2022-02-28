import math

from snake_game import SnakeGameAI


class Individual:
    def __init__(self, game: SnakeGameAI, id, number_of_individuals, screen_w, screen_h, display_padding):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.score = 0
        self.id = id
        self.number_of_individuals = number_of_individuals
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.display_padding = display_padding
        self.set_display()

    def set_display(self):
        square = math.ceil(math.sqrt(self.number_of_individuals))
        game_column = self.id % square
        game_row = self.id // square
        self.game_w = self.screen_w // square - self.display_padding
        self.game_h = self.screen_h // square - self.display_padding
        self.game_x = game_column * \
            (self.game_w + self.display_padding) + self.display_padding
        self.game_y = game_row * (self.game_h + self.display_padding) + \
            self.display_padding
        
    def __repr__(self):
        return f'Id {self.id} Score {self.score} X {self.game_x} Y {self.game_y}'
