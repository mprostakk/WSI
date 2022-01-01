import random
from random import randrange
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygame

from action import Action
from gamemap import GameMap
from helpers import MapPygame, open_map
from my_types import Point


class TurnException(Exception):
    """Maximum number of turns, can not find solution"""


def create_q_table(state_len: int, action_len: int) -> np.array:
    return np.zeros((state_len, action_len))


def should_explore(epsilon: float) -> bool:
    return random.uniform(0, 1) < epsilon


def take_action_from_q_table(q_tab: np.array, state_index: int) -> Action:
    actions_from_state = q_tab[state_index]
    action_index = actions_from_state.argmax()
    return Action(action_index + 1)


def train(
    game_map: GameMap,
    episodes: int = 10000,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.6,
    max_number_of_turns: int = 10000,
) -> np.array:
    number_of_states = game_map.width * game_map.length
    number_of_actions = 4
    q_table = create_q_table(number_of_states, number_of_actions)
    all_steps = []

    for episode in range(1, episodes):
        game_map.reset()
        state_index = game_map.convert_point_to_state_index(game_map.start_point)
        finished = False

        if episode % 10 == 0:
            print(all_steps[-1], episode)

        steps = 0
        while not finished:
            if steps > max_number_of_turns:
                finished = True
                continue

            if should_explore(epsilon):
                action = Action.random()
            else:
                action = take_action_from_q_table(q_table, state_index)

            step = game_map.move(action)
            old_value = q_table[state_index][action.value - 1]
            next_max = q_table[step.next_state_index].max()

            new_value = (1 - learning_rate) * old_value + learning_rate * (
                step.reward + gamma * next_max
            )
            q_table[state_index, action.value - 1] = new_value

            state_index = step.next_state_index
            finished = step.finished
            steps += 1

        all_steps.append(steps)

    return q_table, all_steps


def play(map_pygame: MapPygame, game_map: GameMap, q_table: np.array) -> None:
    game_map.reset()
    finished = False
    state_index = game_map.convert_point_to_state_index(game_map.start_point)
    reward = 0
    steps = 0
    while not finished:
        action = take_action_from_q_table(q_table, state_index)

        step = game_map.move(action)
        reward += step.reward
        state_index = step.next_state_index
        finished = step.finished

        print(action)
        print(reward)
        steps += 1
        game_map.render()

        map_pygame.draw_map(game_map, q_table)
        pygame.time.wait(200)

    print(f"Finished with {steps} steps")


def play_random(game_map: GameMap) -> None:
    game_map.reset()
    finished = False
    reward = 0
    steps = 0
    while not finished:
        action = Action.random()
        step = game_map.move(action)

        reward += step.reward
        print(action)
        print(reward)
        game_map.render()

        steps += 1
        finished = step.finished

    print(f"Finished with {steps} steps")


def plot_steps(steps: List[int], learning_rate: float, epsilon: float, gamma: float):
    # plt.yscale('log')
    plt.plot(steps)

    plt.title(f"Lr={learning_rate}, epsilon={epsilon}, gamma={gamma}")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.show()


def random_generate_map(
    length: int, width: int, start_point: Optional[Point] = None, end_point: Optional[Point] = None
):
    m = [["-"] * width for _ in range(length)]

    for row in m:
        for i in range(len(row)):
            if should_explore(0.4):
                row[i] = "B"

    if start_point is None:
        start_point = (randrange(0, length), randrange(0, width))
    if end_point is None:
        end_point = (randrange(0, length), randrange(0, width))

    m[start_point[0]][start_point[1]] = "S"
    m[end_point[0]][end_point[1]] = "E"

    return m


def check_if_solution_exists_for_map(m):
    pass


def main():
    game_map = open_map("maps/3.txt")

    # play_random(game_map)
    learning_rate = 0.1
    epsilon = 0.1
    gamma = 1.0

    try:
        q_table, steps = train(
            game_map,
            episodes=1000,
            learning_rate=learning_rate,
            epsilon=epsilon,
            gamma=gamma,
            max_number_of_turns=10000,
        )
    except TurnException:
        print("Error")

    plot_steps(steps, learning_rate, epsilon, gamma)

    pygame.init()
    pygame.font.init()
    map_pygame = MapPygame()

    play(map_pygame, game_map, q_table)

    pygame.time.wait(10000)
    pygame.quit()


if __name__ == "__main__":
    main()
