import numpy as np
from numpy import random
from schemas import Population


def mutation(population: Population, probability: float = 0.2):
    new_population = []

    for individual in population:
        if random.uniform(0, 1) >= probability:
            new_population.append(individual)
            continue

        selected_0 = np.where(individual == 0)
        selected_1 = np.where(individual == 1)

        random_index_0 = np.random.choice(selected_0[0])
        random_index_1 = np.random.choice(selected_1[0])

        new_individual = individual.copy()
        new_individual[random_index_0] = 1
        new_individual[random_index_1] = 0

        new_population.append(new_individual)

    return new_population
