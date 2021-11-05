import random

from board import Board, get_wolf_boards, get_sheep_boards


def min_max(board: Board, d: int, maximizing: bool = True):
    if board.is_terminal() or d == 0:
        return board.heuristic(), None

    if maximizing:
        max_value = -10
        best_board = None

        for new_board in get_wolf_boards(board):
            value, _ = min_max(new_board, d - 1, False)
            if value > max_value:
                max_value = value
                best_board = new_board

        return max_value, best_board
    else:
        min_value = 10
        best_board = None

        for new_board in get_sheep_boards(board):
            value, _ = min_max(new_board, d - 1, True)
            if value < min_value:
                min_value = value
                best_board = new_board

        return min_value, best_board


def main():
    wolf_moves = True
    board = Board()
    board.draw()

    while not board.is_terminal():
        if wolf_moves:
            min_max_value, new_board = min_max(board, 8, maximizing=True)

            if new_board is not None:
                board = new_board
            else:
                board = random.choice(get_wolf_boards(board))

        else:
            sheep_boards = get_sheep_boards(board)
            if len(sheep_boards) == 0:
                break
            board = random.choice(sheep_boards)

        board.draw()
        wolf_moves = not wolf_moves


if __name__ == "__main__":
    main()
