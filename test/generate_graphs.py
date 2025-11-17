import numpy as np
import igraph as ig


def generate_random_graph(
    n_vertices: int,
    edge_prob: float = 0.1,
    min_cap: float = 1,
    max_cap: float = 10,
    allow_self_loops=False,
    seed=None
):
    """
    Generates a directed random graph using igraph, with capacities stored in es["capacity"].

    - n_vertices: number of nodes
    - edge_prob: probability of each directed edge (ER graph)
    - min_cap/max_cap: capacity range
    """
    if seed is not None:
        np.random.seed(seed)

    g = ig.Graph(directed=True)
    g.add_vertices(n_vertices)

    edges = []
    capacities = []

    for u in range(n_vertices):
        for v in range(n_vertices):
            if u == v and not allow_self_loops:
                continue
            if np.random.rand() < edge_prob:
                edges.append((u, v))
                cap = np.random.uniform(min_cap, max_cap)
                capacities.append(cap)

    g.add_edges(edges)
    g.es["capacity"] = capacities

    return g
