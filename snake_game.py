import datetime
from typing import Tuple
import pygame
from settings import GAME_WIDTH, GAME_HEIGHT, GAME_TABLE_ROWS, GAME_TABLE_COLUMNS, BLACK, BLOCK_SIZE, BLOCK_DRAW_OFFSET, WHITE, RED
from linked_list import Point, Direction, SnakeLinkedList


class SnakeGameAI:

    def __init__(self, display_gui):
        # init display
        if display_gui:
            self.display = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            self.main_window = pygame.display.get_surface()
            self.font = pygame.font.Font('arial.ttf', 25)
            self.font_symbols = pygame.font.SysFont("DejaVu Sans", 12)
        self.turns = [Direction.DOWN,
                      Direction.LEFT, Direction.RIGHT, Direction.UP]
        self.snake_list = SnakeLinkedList(GAME_TABLE_ROWS, GAME_TABLE_COLUMNS)

        # go_to: strategy to look for corners before foodd
        self.go_to: Point = None

        self.reset()

    def play_step(self, action):
        """receives an action from the agent and updates the game

        Args:
            action (list): list of integers that will determine where to move the snake. [down, left, right, up]

        Returns:
            tuple: reward, game_over, score
        """
        reward = 0
        if not self.game_over:
            self.count_steps += 1

            # 2. move
            self.snake_list.move(action)  # update the head
            self.snake_list.update_board()

            # 3. check if game over
            if self.snake_list.is_collision(self.snake_list.head):
                self.game_over = True
                reward = -10
                return reward, self.game_over, self.score

            # 4. place new food or just move
            if self.snake_list.head == self.snake_list.food:
                self.score += 1
                self.total_steps += self.count_steps
                print(
                    f'{datetime.datetime.now()} Score: {self.score} Steps: {self.count_steps} Total: {self.total_steps}')
                reward = 10
                self.count_steps = 0
                if len(self.snake_list) < GAME_TABLE_COLUMNS*GAME_TABLE_ROWS:
                    self.snake_list.place_food()
                    self.update_go_to()
                else:
                    self.game_over = True
                    return reward, self.game_over, self.score
            elif self.snake_list.head == self.go_to:
                self.update_go_to()

            self._get_distances()
            self._create_dijkstra()

        # 6. return game over and score
        return reward, self.game_over, self.score

    def update_go_to_corner(self, corner: Point):
        """Called when head reached food"""
        current = corner
        working = [current]
        visited = [current]
        while len(working) > 0:
            working = sorted(
                working, key=lambda pt: pt.distance(corner), reverse=True)
            current = working.pop()
            if self.snake_list.grid[current.y][current.x] == self.snake_list.maximum:
                break
            for turn in self.turns:
                move = current + turn
                if not self.snake_list.is_out_of_board(move) and move not in visited:
                    working.append(move)
                    visited.append(move)

        return current

    def update_go_to(self):
        if len(self.snake_list) < self.snake_list.maximum // 4:
            self.go_to = self.snake_list.food
            return

        x_corner_head = 0 if self.snake_list.head.x < GAME_TABLE_COLUMNS//2 else GAME_TABLE_COLUMNS - 1
        y_corner_head = 0 if self.snake_list.head.y < GAME_TABLE_ROWS//2 else GAME_TABLE_ROWS - 1
        corner_head = Point(x_corner_head, y_corner_head)

        x_corner_food = 0 if self.snake_list.food.x < GAME_TABLE_COLUMNS//2 else GAME_TABLE_COLUMNS - 1
        y_corner_food = 0 if self.snake_list.food.y < GAME_TABLE_ROWS//2 else GAME_TABLE_ROWS - 1
        corner_food = Point(x_corner_food, y_corner_food)

        if corner_head == corner_food:
            self.go_to = self.snake_list.food
            return
        self.go_to = self.update_go_to_corner(corner_food)

    def _get_distances(self):
        self.manhattan_distances = [
            (self.snake_list.head + Direction.DOWN).distance(self.snake_list.food),
            (self.snake_list.head + Direction.LEFT).distance(self.snake_list.food),
            (self.snake_list.head + Direction.RIGHT).distance(self.snake_list.food),
            (self.snake_list.head + Direction.UP).distance(self.snake_list.food)
        ]

    def update_ui(self):
        maximum = GAME_TABLE_COLUMNS * GAME_TABLE_ROWS
        self.display.fill(BLACK)

        for index, pt in enumerate(self.snake_list.body):
            index_percent = index/(len(self.snake_list.body)-1)
            blue = (1-index_percent)*255
            green = (index_percent)*255
            color = (0, green, blue)
            color2 = (0, 128, blue)
            self._display_block(color, pt)
            self._display_block(color2, pt, BLOCK_DRAW_OFFSET)
            text = self.font_symbols.render(f'{index}', True, WHITE)
            self.display.blit(
                text, [pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE])

        self._display_block(RED, self.snake_list.food)
        self._display_block((255, 255, 0), self.go_to, BLOCK_DRAW_OFFSET)

        for r, row in enumerate(self.dijkstra):
            for c, value in enumerate(row):
                pt = Point(c, r)
                if value < maximum - 1:
                    text = self.font_symbols.render(f'{value}', True, WHITE)
                    self.display.blit(text, [pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE])

        text = self.font.render(
            "Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

    def _display_block(self, color: Tuple[int, int, int], point: Point, offset: int = 0):
        pygame.draw.rect(self.display, color, pygame.Rect(
            point.x*BLOCK_SIZE + offset, point.y*BLOCK_SIZE + offset, BLOCK_SIZE-2*offset, BLOCK_SIZE-2*offset))

    def reset(self):
        """reset is called at __init__ and by AI agent
        """

        # this will limit the number of moves the snake can make
        # until it is considered game over, default to 100*snake_size
        self.count_steps = 0
        self.total_steps = 0

        self.game_over = False

        # self.snake.reset()
        self.snake_list.reset()
        self.go_to = self.snake_list.food

        self.score = len(self.snake_list)

        self._get_distances()
        self._create_dijkstra()

    def get_board(self, marker: Point = Point(-1, -1)):
        rows = []
        for y in range(GAME_TABLE_ROWS):
            columns = []
            for x in range(GAME_TABLE_COLUMNS):
                pt = Point(x, y)
                if pt == marker:
                    columns.append('MMM')
                elif pt == self.snake_list.food:
                    columns.append('FFF')
                elif pt == self.snake_list.head:
                    columns.append('HHH')
                elif self.snake_list.is_collision(pt):
                    columns.append('SSS')
                elif self.dijkstra[pt.y][pt.x] != -1:
                    columns.append(f'{self.dijkstra[pt.y][pt.x]:03}')

            rows.append("|".join(columns))
        return "\n".join(rows)

    def _get_unicode(self, pt: Point):
        pt = pt.unit()
        if pt.x % 2 == 0 and pt.y % 2:
            code = '\u25F0'
        elif pt.x % 2 and pt.y % 2:
            code = '\u25F1'
        elif pt.x % 2 == 0 and pt.y % 2 == 0:
            code = '\u25F3'
        elif pt.x % 2 and pt.y % 2 == 0:
            code = '\u25F2'
        else:
            code = ' '
        return code

    def _create_dijkstra(self):
        # the distance will never be this value, we can use it as control number
        maximum = self.snake_list.maximum
        # set snake body to one less than maximum, also will never be a distance
        self.dijkstra = [[maximum-1 if self.snake_list.is_collision(Point(c, r)) else maximum for c in range(
            self.snake_list.max_columns)] for r in range(self.snake_list.max_rows)]

        # set head to 0
        self.dijkstra[self.snake_list.head.y][self.snake_list.head.x] = 0
        # reset tail
        self.dijkstra[self.snake_list.tail.y][self.snake_list.tail.x] = maximum

        steps = 0
        neighbors = []
        working = [self.snake_list.head]

        while len(working) > 0:
            working = sorted(
                working, key=lambda pt: pt.distance(self.snake_list.food), reverse=True)
            current = working.pop()
            self.dijkstra[current.y][current.x] = steps
            # if current == self.snake_list.food:
            #     break
            current_direction = current.allowed_directions()
            next_attempt_h = current + \
                (current_direction & Direction.HORIZONTAL)
            next_attempt_v = current + \
                (current_direction & Direction.VERTICAL)
            if not self.snake_list.is_out_of_board(next_attempt_h) and self.dijkstra[next_attempt_h.y][next_attempt_h.x] == maximum:
                self.dijkstra[next_attempt_h.y][next_attempt_h.x] = steps + 1
                neighbors.append(next_attempt_h)
            if not self.snake_list.is_out_of_board(next_attempt_v) and self.dijkstra[next_attempt_v.y][next_attempt_v.x] == maximum:
                self.dijkstra[next_attempt_v.y][next_attempt_v.x] = steps + 1
                neighbors.append(next_attempt_v)

            if len(working) == 0 and len(neighbors) > 0:
                steps += 1
                working += neighbors
                neighbors = []
        self.shortest_dijkstra()
        self.dijkstra_gap()

    def shortest_dijkstra(self):
        maximum = GAME_TABLE_COLUMNS*GAME_TABLE_ROWS

        # target can be the go_to or the tail
        target = self.go_to
        if self.dijkstra[target.y][target.x] >= maximum - 2:
            target = self.snake_list.tail

        while self.dijkstra[target.y][target.x] != 1:
            moves = [target + turn for turn in self.turns]
            moves = sorted(moves, key=lambda pt: pt.distance(
                self.snake_list.head))
            for move in moves:
                if not self.snake_list.is_out_of_board(move):
                    value = self.dijkstra[move.y][move.x]
                    if value == self.dijkstra[target.y][target.x] - 1:
                        break
            target = move

        self.short_dijkstra = [
            int((self.snake_list.head+turn) == target) for turn in self.turns]

    def dijkstra_gap(self):

        # if can't find shortest path to target
        if sum(self.short_dijkstra) == 0:
            return

        # get the suggested turn
        suggested_turn = self.turns[self.short_dijkstra.index(1)]
        control: Point = self.snake_list.head + suggested_turn

        # get the other possible turn
        allowed_directions = self.snake_list.head.allowed_directions()
        other_turn: Point = allowed_directions - suggested_turn

        # start at other turn
        current: Point = self.snake_list.head + other_turn
        # check if other turn is collision
        if self.snake_list.is_collision(current):
            return

        food_dijkstra = 0
        if self.dijkstra[self.snake_list.food.y][self.snake_list.food.x] < GAME_TABLE_COLUMNS*GAME_TABLE_ROWS - 1:
            food_dijkstra = self.dijkstra[self.snake_list.food.y][self.snake_list.food.x]

        # start at other turn
        working = [current]
        visited = [current]
        found_tail = False
        highest_snake_index = 0
        while len(working) > 0:
            working = sorted(
                working, key=lambda pt: pt.distance(self.snake_list.tail), reverse=True)

            current = working.pop()
            if self.snake_list.tail == current:
                found_tail = True
                break

            for turn in self.turns:
                move = current + turn
                if self.snake_list.is_out_of_board(move):
                    continue

                if self.snake_list.is_hit(move) and self.snake_list.tail != move:
                    snake_index = self.snake_list.get_snake_index(move)
                    if snake_index > highest_snake_index:
                        highest_snake_index = snake_index
                elif move not in visited and move != control and self.snake_list.head != move:
                    working.append(move)
                    visited.append(move)

            if len(visited) > len(self.snake_list):
                found_tail = True
                break

        # if the tail was not found, but the steps to get to the food is greater
        # than the number of steps the tail has to do to get there, it is not a gap
        if not found_tail and food_dijkstra > len(self.snake_list) - highest_snake_index:
            found_tail = True

        # if it can't reach the tail, it is a gap
        if not found_tail:
            self.short_dijkstra = [int(other_turn == turn)
                                   for turn in self.turns]
