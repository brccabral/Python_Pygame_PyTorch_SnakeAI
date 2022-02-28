class Individual:
    def __init__(self, game, id):
        self.game = game
        self.game_over = False
        self.reward = 0
        self.scode = 0
        self.id = id
