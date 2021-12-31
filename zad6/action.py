import random
from enum import Enum

from my_types import Point


class Action(Enum):
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4

    @staticmethod
    def random():
        return random.choice(list(Action))

    @staticmethod
    def get_new_point(point: Point, action):
        if action == Action.UP:
            return point[0] - 1, point[1]
        elif action == Action.RIGHT:
            return point[0], point[1] + 1
        elif action == Action.DOWN:
            return point[0] + 1, point[1]
        else:
            return point[0], point[1] - 1
