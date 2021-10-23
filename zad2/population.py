import numpy as np
from schemas import Population


def init_population(
    number_of_population: int, number_of_vertices: int, number_of_covered_vertices: int
) -> Population:
    population = []
    for i in range(number_of_population):
        individual = np.zeros((number_of_vertices,), dtype=int)
        individual[:number_of_covered_vertices] = 1
        np.random.shuffle(individual)
        population.append(individual)

    return population
