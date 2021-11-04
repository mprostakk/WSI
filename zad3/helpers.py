def init_board():
    rows = []
    for _ in range(8):
        rows.append([' ' for _ in range(8)])
    return rows


def print_board(board):
    s = ''
    for row in board:
        s = '|'.join([f' {x} ' for x in row])
        print('-' * (len(s) + 2))
        print('|' + s + '|')

    print('-'*(len(s) + 3))
    print()


def setup_board(board):
    first_row = board[0]
    last_row = board[-1]

    for i in range(1, 8, 2):
        first_row[i] = 'O'

    last_row[0] = 'W'
