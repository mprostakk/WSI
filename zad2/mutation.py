import numpy as np
from numpy import random
from schemas import Population


def mutation(population: Population, probability: float = 0.2, amount_probability: float = 1.0):
    new_population = []

    for individual in population:
        if random.uniform(0, 1) >= probability:
            new_population.append(individual)
            continue

        selected_0 = np.where(individual == 0)[0]
        selected_1 = np.where(individual == 1)[0]

        max_number_of_mutations = min(len(selected_0), len(selected_1))

        mu, sigma = 0, amount_probability
        random_number_of_mutations = np.random.normal(mu, sigma)
        random_number_of_mutations = int(np.round(np.abs(random_number_of_mutations)) + 1)
        if random_number_of_mutations > max_number_of_mutations:
            random_number_of_mutations = max_number_of_mutations

        random_index_0 = np.random.choice(selected_0, random_number_of_mutations, replace=False)
        random_index_1 = np.random.choice(selected_1, random_number_of_mutations, replace=False)

        new_individual = individual.copy()

        for i in range(random_number_of_mutations):
            index_0 = random_index_0[i]
            index_1 = random_index_1[i]

            new_individual[index_0] = 1
            new_individual[index_1] = 0

        if np.sum(new_individual) != np.sum(individual):
            raise NotImplementedError()

        new_population.append(new_individual)

    return new_population
