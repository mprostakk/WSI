import random
from collections import defaultdict

import pygame
from ai import SheepAI, WolfAI
from board import Board, get_sheep_boards, get_wolf_boards
from board_pygame import BoardPygame


def main():
    wolf_moves = True
    sheep_is_ai = True
    board = Board()
    board.draw()

    pygame.init()

    board_pygame = BoardPygame()

    while not board.is_terminal():
        if wolf_moves:
            wolf_ai = WolfAI()
            d = defaultdict(list)

            for wolf_move in get_wolf_boards(board):
                min_max_value = wolf_ai.min_max(wolf_move, 1, maximizing=False)
                d[min_max_value].append(wolf_move)

            new_board = random.choice(d[max(d)])
            board = new_board
            print("Wolf: ", max(d))

        else:
            if not sheep_is_ai:
                sheep_boards = get_sheep_boards(board)
                if len(sheep_boards) == 0:
                    break
                board = random.choice(sheep_boards)
            else:
                sheep_ai = SheepAI()
                d = defaultdict(list)

                for sheep_move in get_sheep_boards(board):
                    min_max_value = sheep_ai.min_max(sheep_move, 6, maximizing=True)
                    d[min_max_value].append(sheep_move)

                new_board = random.choice(d[min(d)])
                board = new_board

                print("Sheep: ", min(d))

        # board.draw()
        pygame.time.wait(200)
        board_pygame.draw_board(board)

        wolf_moves = not wolf_moves

    pygame.quit()

    if board.did_wolf_win():
        print("Wolf won")
    elif board.did_sheep_win():
        print("Sheep won")
    else:
        print("Draw?")


if __name__ == "__main__":
    main()
