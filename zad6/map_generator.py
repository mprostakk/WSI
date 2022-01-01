import random
from copy import deepcopy
from random import randrange
from typing import Optional

from gamemap import GameMap
from my_types import Point


def random_generate_map(
    length: int,
    width: int,
    start_point: Optional[Point] = None,
    end_point: Optional[Point] = None,
    epsilon: float = 0.5,
) -> GameMap:
    game_map = _random_generate_map(length, width, start_point, end_point, epsilon)
    while check_if_solution_exists_for_map(deepcopy(game_map)) is False:
        game_map = _random_generate_map(length, width, start_point, end_point, epsilon)

    return game_map


def _random_generate_map(
    length: int,
    width: int,
    start_point: Optional[Point] = None,
    end_point: Optional[Point] = None,
    epsilon: float = 0.5,
) -> GameMap:
    m = [["-"] * width for _ in range(length)]

    for row in m:
        for i in range(len(row)):
            if random.uniform(0, 1) < epsilon:
                row[i] = "B"

    if start_point is None:
        start_point = (randrange(0, length), randrange(0, width))
    if end_point is None:
        end_point = (randrange(0, length), randrange(0, width))

    m[start_point[0]][start_point[1]] = "S"
    m[end_point[0]][end_point[1]] = "E"

    return GameMap(
        length=length,
        width=width,
        start_point=start_point,
        end_point=end_point,
        current_point=start_point,
        state=m,
    )


def check_if_solution_exists_for_map(game_map: GameMap) -> bool:
    if game_map.check_if_won(game_map.current_point):
        return True

    game_map.state[game_map.current_point[0]][game_map.current_point[1]] = "X"

    points = game_map.get_new_points()
    for point in points:
        if game_map.state[point[0]][point[1]] in ["-", "E"]:
            game_map.current_point = point
            if check_if_solution_exists_for_map(game_map):
                return True

    return False
