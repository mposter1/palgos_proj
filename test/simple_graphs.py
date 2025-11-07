import igraph as ig

def graph_0():
    # Construct a graph with 5 vertices
    g = ig.Graph(
        6,
        [(3, 2), (3, 4), (2, 1), (4, 1), (4, 5), (1, 0), (5, 0)],
        directed=True
    )
    g.es["capacity"] = [7, 8, 1, 2, 3, 4, 5]
    return g

def graph_1():
    # Construct a graph with 4 vertices: s=0, a=1, b=2, t=3
    g = ig.Graph(
        4,
        [(0,1), (1,3), (0,2), (2,3)],
        directed=True
    )
    g.es["capacity"] = [3, 3, 2, 2]
    return g

def graph_2():
    # Construct a graph with 5 vertices: s=0, a=1, b=2, c=3, t=4
    g = ig.Graph(
        5,
        [(0,1), (0,2), (1,3), (2,3), (3,4)],
        directed=True
    )
    g.es["capacity"] = [10, 10, 4, 6, 15]
    return g

def graph_3():
    # Construct a graph with 5 vertices: s=0, a=1, b=2, c=3, t=4
    g = ig.Graph(
        5,
        [(0,1), (0,2), (1,4), (2,4), (1,3), (2,3), (3,4)],
        directed=True
    )
    g.es["capacity"] = [5, 5, 3, 4, 2, 4, 10]
    return g

def graph_4():
    # Construct a graph with 3 vertices: s=0, a=1, t=2
    g = ig.Graph(
        3,
        [(0,1), (1,2)],
        directed=True
    )
    g.es["capacity"] = [100, 1]
    return g

def graph_5():
    # Construct a graph with 4 vertices: s=0, a=1, b=2, t=3
    g = ig.Graph(
        4,
        [(0,1), (0,2), (1,2), (1,3), (2,3)],
        directed=True
    )
    g.es["capacity"] = [4, 2, 2, 2, 4]
    return g

def graph_6():
    # Construct a graph with 7 vertices: s=0, a=1, b=2, c=3, d=4, e=5, t=6
    g = ig.Graph(
        7,
        [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5), (5,6)],
        directed=True
    )
    g.es["capacity"] = [8, 8, 4, 4, 3, 1, 10]
    return g

def graph_7():
    # Construct a graph with 6 vertices: s=0, a=1, b=2, c=3, d=4, t=5
    g = ig.Graph(
        6,
        [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5), (3,4), (4,3)],
        directed=True
    )
    g.es["capacity"] = [2, 2, 2, 2, 1, 3, 1, 1]
    return g

def graph_8():
    # Construct a graph with 4 vertices: s=0, a=1, b=2, t=3
    g = ig.Graph(
        4,
        [(0,1), (0,2), (1,2), (1,3), (2,3)],
        directed=True
    )
    g.es["capacity"] = [3, 3, 5, 2, 1]
    return g
