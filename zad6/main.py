import random

import numpy as np

from action import Action
from gamemap import GameMap
from helpers import open_map


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
    epochs: int = 10000,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.6,
) -> np.array:
    number_of_states = game_map.width * game_map.length
    number_of_actions = 4
    q_table = create_q_table(number_of_states, number_of_actions)

    for epoch in range(epochs):
        game_map.reset()
        state_index = game_map.convert_point_to_state_index(game_map.start_point)
        finished = False

        if epoch % 100 == 0:
            print(epoch)

        while not finished:
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

    return q_table


def play(game_map: GameMap, q_table: np.array) -> None:
    game_map.reset()
    finished = False
    state_index = 0
    reward = 0
    epochs = 0
    while not finished:
        action = take_action_from_q_table(q_table, state_index)

        step = game_map.move(action)
        reward += step.reward
        state_index = step.next_state_index
        finished = step.finished

        print(action)
        print(reward)
        epochs += 1
        game_map.render()

    print(f"Finished with {epochs} epochs")


def play_random(game_map: GameMap) -> None:
    game_map.reset()
    finished = False
    reward = 0
    epochs = 0
    while not finished:
        action = Action.random()
        step = game_map.move(action)

        reward += step.reward
        print(action)
        print(reward)
        game_map.render()

        epochs += 1
        finished = step.finished

    print(f"Finished with {epochs} epochs")


def main():
    game_map = open_map("maps/1.txt")
    # play_random(game_map)

    q_table = train(game_map)
    play(game_map, q_table)


if __name__ == "__main__":
    main()
