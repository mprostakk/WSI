from typing import List
from copy import copy

from helpers import init_board, print_board


class Board:
    def __init__(self) -> None:
        self.wolf = (7, 0)
        self.sheep = [(0, 1), (0, 3), (0, 5), (0, 7)]

    def draw(self) -> None:
        board = init_board()
        board[self.wolf[0]][self.wolf[1]] = 'W'

        for sheep in self.sheep:
            board[sheep[0]][sheep[1]] = 'O'

        print_board(board)

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

    def get_new_moves_from_point(self, point):
        new_points = generate_new_points_from_point(point)
        return [p for p in new_points if self.check_if_move_possible(p)]


def generate_new_points_from_point(point) -> List:
    return [
        (point[0] + 1, point[1] + 1),
        (point[0] + 1, point[1] - 1),
        (point[0] - 1, point[1] + 1),
        (point[0] - 1, point[1] - 1),
    ]


def get_wolf_boards(board: Board) -> List[Board]:
    new_wolf_points = board.get_new_moves_from_point(board.wolf)
    new_boards = []
    for new_wolf_point in new_wolf_points:
        new_board = copy(board)
        new_board.wolf = new_wolf_point
        new_boards.append(
            new_board
        )

    return new_boards


def get_sheep_boards(board: Board) -> List[Board]:
    new_boards = []
    for index, sheep in enumerate(board.sheep):
        new_sheep_points = board.get_new_moves_from_point(sheep)
        for new_sheep_point in new_sheep_points:
            new_board = copy(board)
            new_board.sheep = copy(new_board.sheep)
            new_board.sheep[index] = new_sheep_point
            new_boards.append(
                new_board
            )

    return new_boards
