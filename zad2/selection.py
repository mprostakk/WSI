import random

from schemas import Population
from score import individual_score


def selection(population: Population, graph):
    # Tournament selection

    # TODO - losowanie bez zwracania
    # k = 2

    new_population = []
    number_of_tournaments = len(population)

    for i in range(number_of_tournaments):
        random_individual_1 = random.choice(population)
        random_individual_2 = random.choice(population)

        if individual_score(random_individual_1, graph) <= individual_score(
            random_individual_2, graph
        ):
            new_population.append(random_individual_1)
        else:
            new_population.append(random_individual_2)

    return new_population
