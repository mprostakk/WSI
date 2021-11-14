import math

from board import Board, get_sheep_boards, get_wolf_boards


class AI:
    def min_max(self, board: Board, d: int, maximizing: bool = True):
        if board.is_terminal() or d == 0:
            return self.heuristic(board)

        if maximizing:
            max_value = -math.inf

            for new_board in get_wolf_boards(board):
                value = self.min_max(new_board, d - 1, False)
                if value > max_value:
                    max_value = value

            return max_value
        else:
            min_value = math.inf

            for new_board in get_sheep_boards(board):
                value = self.min_max(new_board, d - 1, True)
                if value < min_value:
                    min_value = value

            return min_value

    def heuristic(self, board):
        raise NotImplementedError()


class WolfAI(AI):
    def heuristic(self, board):
        return board.heuristic()


class SheepAI(AI):
    def heuristic(self, board):
        return board.heuristic_sheep()
