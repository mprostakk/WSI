import random
from copy import copy
from typing import List

from helpers import init_board, print_board2


class Board:
    def __init__(self) -> None:
        # Real game points
        x_wolf = random.randint(0, 3) * 2
        self.wolf = (7, x_wolf)
        self.sheep = [(0, 1), (0, 3), (0, 5), (0, 7)]

    def draw(self) -> None:
        board = init_board()
        board[self.wolf[0]][self.wolf[1]] = "W"

        for sheep in self.sheep:
            board[sheep[0]][sheep[1]] = "O"

        print_board2(board)

    def check_if_move_possible(self, new_point) -> bool:
        if new_point[0] < 0 or new_point[0] >= 8:
            return False

        if new_point[1] < 0 or new_point[1] >= 8:
            return False

        if new_point == self.wolf:
            return False

        for sheep in self.sheep:
            if sheep == new_point:
                return False

        return True

    def get_wolf_new_moves_from_point(self, point):
        new_points = generate_new_points_from_point(point)
        return [p for p in new_points if self.check_if_move_possible(p)]

    def get_sheep_new_moves_from_point(self, point):
        new_points = generate_new_points_going_down(point)
        return [p for p in new_points if self.check_if_move_possible(p)]

    def did_wolf_win(self):
        sheep_moves = sum([len(self.get_sheep_new_moves_from_point(x)) for x in self.sheep])
        return self.wolf[0] == 0 or sheep_moves == 0

    def did_sheep_win(self):
        wolf_moves = self.get_wolf_new_moves_from_point(self.wolf)
        return len(wolf_moves) == 0

    def is_terminal(self):
        return self.did_wolf_win() or self.did_sheep_win()

    def heuristic(self):
        x = 0
        if self.did_wolf_win():
            x = 1000
        if self.did_sheep_win():
            x = -1000

        wolf_points = 7 - self.wolf[0]
        wolf_points *= 100

        return wolf_points + x

    def heuristic_sheep(self):
        x = 0
        if self.did_wolf_win():
            x = 1000
        if self.did_sheep_win():
            x = -1000

        points = 7 - self.wolf[0]
        points *= 10

        # sheep_points = sum([x[0] for x in self.sheep])
        lowest_sheep = max([x[0] for x in self.sheep])

        if self.wolf[0] < lowest_sheep:
            points += 300

        wolf_new_points = generate_new_points_going_up(self.wolf)
        for w in wolf_new_points:
            if w in self.sheep:
                points -= 300

        return points + x


def generate_new_points_from_point(point) -> List:
    return [
        (point[0] + 1, point[1] + 1),
        (point[0] + 1, point[1] - 1),
        (point[0] - 1, point[1] + 1),
        (point[0] - 1, point[1] - 1),
    ]


def generate_new_points_going_down(point):
    return [
        (point[0] + 1, point[1] + 1),
        (point[0] + 1, point[1] - 1),
    ]


def generate_new_points_going_up(point):
    return [
        (point[0] - 1, point[1] + 1),
        (point[0] - 1, point[1] - 1),
    ]


def get_wolf_boards(board: Board) -> List[Board]:
    new_wolf_points = board.get_wolf_new_moves_from_point(board.wolf)
    new_boards = []
    for new_wolf_point in new_wolf_points:
        new_board = copy(board)
        new_board.wolf = new_wolf_point
        new_boards.append(new_board)

    return new_boards


def get_sheep_boards(board: Board) -> List[Board]:
    new_boards = []
    for index, sheep in enumerate(board.sheep):
        new_sheep_points = board.get_sheep_new_moves_from_point(sheep)
        for new_sheep_point in new_sheep_points:
            new_board = copy(board)
            new_board.sheep = copy(new_board.sheep)
            new_board.sheep[index] = new_sheep_point
            new_boards.append(new_board)

    return new_boards
