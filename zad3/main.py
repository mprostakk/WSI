from board import Board, get_wolf_boards, get_sheep_boards


def is_terminal(board):
    raise NotImplementedError()


def heuristic(board):
    raise NotImplementedError()


def each_move(board, is_wolf: bool):
    if is_wolf:
        return each_move_for_wolf(board)
    else:
        return each_move_for_sheep(board)


def each_move_for_wolf(board):
    new_boards = []


def each_move_for_sheep(board):
    pass


def min_max(board, d, maximizing: bool):
    if is_terminal(board) or d == 0:
        return heuristic(board)

    if maximizing:
        max_value = -10_000
        for new_board in each_move(board, maximizing):
            value = min_max(new_board, d + 1, False)
            max_value = max(max_value, value)

        return max_value
    else:
        min_value = 10_000
        for new_board in each_move(board):
            value = min_max(new_board, d + 1, True)
            min_value = min(min_value, value)

        return min_value


def main():
    board = Board()

    new_boards = get_sheep_boards(board)

    # new_boards = get_wolf_boards(board)
    # board = new_boards[0]
    # board.draw()
    # new_boards = get_wolf_boards(board)

    for b in new_boards:
        b.draw()


if __name__ == "__main__":
    main()
