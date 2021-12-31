from copy import deepcopy
from dataclasses import dataclass

from action import Action
from my_types import Point, State
from step import Step


@dataclass
class GameMap:
    width: int
    length: int
    start_point: Point
    end_point: Point
    current_point: Point
    state: State

    def reset(self):
        self.current_point = self.start_point

    def check_move(self, point):
        if point[0] < 0 or point[1] < 0:
            return False

        if point[0] >= self.length or point[1] >= self.width:
            return False

        if self.state[point[0]][point[1]] != "B":
            return True

        return False

    def check_if_won(self, point):
        return self.state[point[0]][point[1]] == "E"

    def convert_point_to_state_index(self, point) -> int:
        return point[0] * self.width + point[1]

    def move(self, action: Action):
        new_point = Action.get_new_point(self.current_point, action)
        old_state_index = self.convert_point_to_state_index(self.current_point)

        if self.check_move(new_point):
            new_state_index = self.convert_point_to_state_index(new_point)

            reward = -1
            finished = False
            if self.check_if_won(new_point):
                reward = 100
                finished = True

            self.current_point = new_point
            return Step(
                reward=reward,
                next_state_index=new_state_index,
                old_state_index=old_state_index,
                finished=finished,
            )
        else:
            return Step(
                reward=-10,
                next_state_index=old_state_index,
                old_state_index=old_state_index,
                finished=False,
            )

    def render(self):
        state = deepcopy(self.state)
        state[self.current_point[0]][self.current_point[1]] = "X"
        for row in state:
            print(row)
        print()
