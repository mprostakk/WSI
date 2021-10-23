import logging
import sys

import networkx as nx
from draw import draw_best
from mutation import mutation
from population import init_population
from score import rate_population, rating
from selection import selection

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


"""
Example Problem

Given a graph:

0 - (1) - 2
     | 
    (3) - 4

The vertices 1 and 3 are selected.

i: 0 1 2 3 4
p: 0 1 0 1 0

The score is 0, which means we have found an example solution for covering all edges 

"""


def genetic(
    graph,
    vertices: int,
    number_of_generations: int,
    number_of_population: int,
    mutation_probability: float,
    number_of_covered_vertices: int,
):
    population = init_population(
        number_of_population=number_of_population,
        number_of_vertices=vertices,
        number_of_covered_vertices=number_of_covered_vertices,
    )

    ratings = []

    generation = 0
    best_score = rating(population, graph)
    best_population = population

    while generation < number_of_generations:
        selected_population = selection(population, graph)
        mutated_population = mutation(selected_population)
        score = rating(mutated_population, graph)

        population = mutated_population

        ratings.append(rate_population(population, graph))

        sys.stdout.write("\rGeneration %d" % generation)
        sys.stdout.flush()

        if best_score == 0:
            print("\nFound a solution!")
            break

        if score < best_score:
            best_score = score
            best_population = population
            print(f"\nNew best Score: {score}")

        generation += 1

    print("")
    draw_best(best_population, graph)


def draw_scatter():
    pass
    # print(ratings)
    # x = []
    # y = []
    # i = 0
    # for population_rating in ratings:
    #     for r in population_rating:
    #         x.append(i)
    #         y.append(r)
    #     i += 1
    # plt.scatter(x, y)
    # plt.show()


def main():
    # graph = numpy.array([
    #     [0, 1, 0, 0, 0],
    #     [1, 0, 1, 1, 0],
    #     [0, 1, 0, 0, 0],
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 0, 1, 0],
    # ])
    # g = nx.fast_gnp_random_graph(
    #     NUMBER_OF_VERTICES,
    #     0.3,
    #     seed=1
    # )
    g = nx.grid_2d_graph(5, 5)
    graph = nx.convert_matrix.to_numpy_array(g)

    vertices = 25
    number_of_generations = 1000
    number_of_population = 20
    mutation_probability = 0.4
    number_of_covered_vertices = 12

    genetic(
        graph=graph,
        vertices=vertices,
        number_of_generations=number_of_generations,
        number_of_population=number_of_population,
        mutation_probability=mutation_probability,
        number_of_covered_vertices=number_of_covered_vertices,
    )


if __name__ == "__main__":
    main()
