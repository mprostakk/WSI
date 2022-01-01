import math

import numpy as np
import pygame
from action import Action
from gamemap import GameMap
from my_types import Point


def open_map(filename: str) -> GameMap:
    state = []
    start_point = (-1, -1)
    end_point = (-1, -1)
    width = 0
    length = 0

    with open(filename, "r") as map_file:
        for index, line in enumerate(map_file):
            width = len(line)
            length += 1
            state.append([x for x in line if x != "\n"])

            if (start_point_index := line.find("S")) != -1:
                start_point = (index, start_point_index)

            if (end_point_index := line.find("E")) != -1:
                end_point = (index, end_point_index)

    return GameMap(
        start_point=start_point,
        end_point=end_point,
        state=state,
        current_point=start_point,
        width=width,
        length=length,
    )


class MapPygame:
    def __init__(self):
        self.screen_size = 800
        self.dark_grey, self.black, self.white, self.green, self.red = (
            (25, 25, 25),
            (20, 20, 20),
            (200, 200, 200),
            (0, 255, 0),
            (255, 0, 0),
        )
        self.size = self.screen_size // 8
        self.game_display = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.font = pygame.font.SysFont("helvetica", 30)

    def draw_map(self, game_map: GameMap, q_table: np.array) -> None:
        self.game_display.fill(self.black)

        for x in range(game_map.width):
            for y in range(x % 2, game_map.length, 2):
                pygame.draw.rect(
                    self.game_display,
                    self.dark_grey,
                    (x * self.size, y * self.size, self.size, self.size),
                )

        s = self.size
        s2 = self.size // 2

        pygame.draw.rect(
            self.game_display,
            self.green,
            (game_map.start_point[1] * s, game_map.start_point[0] * s, s, s),
            2,
        )

        pygame.draw.rect(
            self.game_display,
            self.red,
            (game_map.end_point[1] * s, game_map.end_point[0] * s, s, s),
            2,
        )

        current_point = game_map.current_point

        pygame.draw.rect(
            self.game_display,
            self.green,
            (current_point[1] * s + s2 // 2, current_point[0] * s + s2 // 2, s2, s2),
        )

        wall_points = game_map.get_wall_points()
        for wall_point in wall_points:
            pygame.draw.rect(
                self.game_display,
                self.white,
                (wall_point[1] * s, wall_point[0] * s, s, s),
            )

        for i in range(game_map.width):
            for j in range(game_map.length):
                p = (j, i)
                state_index = game_map.convert_point_to_state_index(p)
                action = Action(q_table[state_index].argmax() + 1)
                draw_arrow_2(
                    self.game_display, (150, 150, 150), (i * s + s // 2, j * s + s // 2), action
                )

        pygame.display.update()


def draw_arrow_2(screen, color, position: Point, action: Action):
    p1 = position
    p2 = position
    flip = False

    x, y = 0, 0

    if action == Action.UP:
        y = 10
        flip = True
    elif action == Action.DOWN:
        y = 10
    elif action == action.LEFT:
        x = 10
        flip = True
    else:
        x = 10

    p1 = (p1[0] - x, p1[1] - y)
    p2 = (p2[0] + x, p2[1] + y)

    if flip:
        p1, p2 = p2, p1

    draw_arrow(screen, color, p1, p2)


def draw_arrow(screen, color, start, end):
    pygame.draw.line(screen, color, start, end, 2)
    rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
    x = 10
    pygame.draw.polygon(
        screen,
        color,
        (
            (
                end[0] + x * math.sin(math.radians(rotation)),
                end[1] + x * math.cos(math.radians(rotation)),
            ),
            (
                end[0] + x * math.sin(math.radians(rotation - 120)),
                end[1] + x * math.cos(math.radians(rotation - 120)),
            ),
            (
                end[0] + x * math.sin(math.radians(rotation + 120)),
                end[1] + x * math.cos(math.radians(rotation + 120)),
            ),
        ),
    )
