import matplotlib.pyplot as plt
import networkx as nx
from schemas import Population
from score import individual_score


def draw_best(population: Population, graph):
    min_score = individual_score(population[0], graph)
    min_individual = population[0]

    for individual in population:
        score = individual_score(individual, graph)
        if score < min_score:
            min_score = score
            min_individual = individual

    G = nx.from_numpy_matrix(graph)
    color_map = []

    for x in min_individual:
        if x == 0:
            color_map.append("blue")
        else:
            color_map.append("red")

    for i, node in enumerate(G):
        edges = G.edges(node)
        if min_individual[i] == 1:
            for edge in edges:
                G[edge[0]][edge[1]]["color"] = "black"

    edge_color_list = [G[e[0]][e[1]].get("color", "red") for e in G.edges()]

    nx.draw(G, node_color=color_map, with_labels=True, edge_color=edge_color_list)
    plt.show()

    print(f"Finished with best score {min_score}")
