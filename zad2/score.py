import numpy as np
from schemas import Population


def individual_score(individual, graph) -> int:
    g = graph.copy()
    for i, x in enumerate(individual):
        if x == 1:
            g[i, :] = 0
            g[:, i] = 0

    return int(g.sum() // 2)


def rating(population: Population, graph):
    scores = list()

    for individual in population:
        score = individual_score(individual, graph)
        scores.append(score)

    return np.min(scores)


def rate_population(population: Population, graph):
    scores = list()

    for individual in population:
        score = individual_score(individual, graph)
        scores.append(score)

    return scores
