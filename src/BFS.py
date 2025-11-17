import cupy as cp

from collections import deque

def bfs_serial(adj, s, t):
    """
    adj : np.array (n x n) dense adjacency/capacity matrix
    s   : source node
    t   : target node

    Returns True if there is a path from source 's' to sink 't' in the graph.
    """
    n = adj.shape[0]

    # Create a visited array and mark all vertices as not visited
    visited = [False] * n

    # Create a queue, enqueue source vertex and mark source vertex
    # as visited
    q = deque()
    q.append(s)
    visited[s] = True
    # parent[s] = -1

    # Standard BFS Loop
    while q:
        u = q.popleft()
        for v in range(n):
            if visited[v] is False and adj[u][v] > 0:
                q.append(v)
                # parent[v] = u
                visited[v] = True

    # If we reached sink in BFS starting from source, then return
    # True, else False
    return visited[t]


def bfs_vectorized(adj, s, t):
    """
    adj : cp.ndarray (V x V) dense adjacency/capacity matrix (GPU)
    s   : source node
    t   : target node

    Returns True if there is a path from source 's' to sink 't' in the graph.
    """

    V = adj.shape[0]

    # Convert adjacency to boolean connectivity mask on GPU
    adj_bool = (adj > 0)

    # GPU boolean vectors
    visited = cp.zeros(V, dtype=cp.bool_)
    frontier = cp.zeros(V, dtype=cp.bool_)

    visited[s] = True
    frontier[s] = True

    while frontier.any():
        # Matrix multiply to propagate frontier one step
        # frontier is shape (V,), so reshape for matrix multiplication
        neighbors = frontier.astype(cp.int8) @ adj_bool
        neighbors = neighbors.astype(cp.bool_)

        # Mask out visited nodes
        new_frontier = neighbors & (~visited)

        # If sink reached
        if new_frontier[t]:
            return True

        visited |= new_frontier
        frontier = new_frontier

    return visited[t]
