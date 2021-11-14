import random

import pygame

from ai import WolfAI, SheepAI
from board import Board, get_wolf_boards, get_sheep_boards
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
            d = {}
            for wolf_move in get_wolf_boards(board):
                min_max_value = wolf_ai.min_max(wolf_move, 2, maximizing=False)
                d[min_max_value] = wolf_move

            new_board = d[max(d)]
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
                d = {}
                for sheep_move in get_sheep_boards(board):
                    min_max_value = sheep_ai.min_max(sheep_move, 6, maximizing=True)
                    d[min_max_value] = sheep_move

                new_board = d[min(d)]
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
