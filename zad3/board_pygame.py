import pygame
from board import Board


class BoardPygame:
    def __init__(self):
        self.screen_size = 800
        self.white, self.black, self.wolf, self.sheep = (
            (200, 200, 200),
            (50, 50, 50),
            (255, 0, 0),
            (0, 255, 0),
        )
        self.size = self.screen_size // 8
        self.game_display = pygame.display.set_mode((800, 800))

    def draw_board(self, board: Board):
        self.game_display.fill(self.white)

        for x in range(8):
            for y in range(x % 2, 8, 2):
                pygame.draw.rect(
                    self.game_display,
                    self.black,
                    (x * self.size, y * self.size, self.size, self.size),
                )

        s = self.size
        s2 = self.size // 2

        pygame.draw.rect(
            self.game_display,
            self.wolf,
            (board.wolf[1] * s + s2 // 2, board.wolf[0] * s + s2 // 2, s2, s2),
        )

        for sheep in board.sheep:
            pygame.draw.rect(
                self.game_display,
                self.sheep,
                (sheep[1] * s + s2 // 2, sheep[0] * s + s2 // 2, s2, s2),
            )

        pygame.display.update()
