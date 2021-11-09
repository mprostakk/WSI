def init_board():
    rows = []
    for _ in range(8):
        rows.append([' ' for _ in range(8)])
    return rows


def print_board2(board):
    s = ''
    for row in board:
        s = '|'.join([f' {x} ' for x in row])
        print('-' * (len(s) + 2))
        print('|' + s + '|')

    print('-'*(len(s) + 3))
    print()


def print_board1(board):
    for row in board:
        s = ''.join(row)
        print(s)

    print('-'*10)
