import numpy as np
import cupy as cp


class dinics_cuda:
    def __init__(self, G):
        self.adj = cp.array(G.get_adjacency(attribute="capacity").data, dtype=cp.float32)
        self.level = cp.empty(self.adj.shape[0])

    def construct_levels(self, s, t) -> bool:
        """
        s   : source node
        t   : target/sink node

        Assigns levels to the vertices in the given graph.
        Returns False is no path exists from source to sink.
        """

        self.level.fill(-1)
        self.level[s] = 0

        current_level = 0

        # Convert adjacency to boolean connectivity mask on GPU
        adj_bool = (self.adj > 0)
        frontier = cp.zeros(self.adj.shape[0], dtype=cp.bool_)
        frontier[s] = True

        while frontier.any():
            # Matrix multiply to propagate frontier one step
            # frontier is shape (V,), so reshape for matrix multiplication
            neighbors = frontier.astype(cp.int8) @ adj_bool
            neighbors = neighbors.astype(cp.bool_)
            new_frontier = neighbors & (self.level == -1)
            self.level[new_frontier] = current_level + 1
            frontier = new_frontier
            current_level += 1

        return self.level[t] != -1

    def send_flow(self, residual, flow, s, t):
        total_flow = 0.0
        stack = [(s, cp.float32('inf'), [])]  # (vertex, path_flow, path)
        ptr = cp.zeros(self.adj.shape[0], dtype=cp.int32)  # tracks next neighbor per vertex

        while stack:
            u, path_flow, path = stack.pop()
            path = path + [u]

            if u == t:
                # Push flow along the path
                min_residual = path_flow
                for i in range(len(path)-1):
                    x, y = path[i], path[i+1]
                    flow[x, y] += min_residual
                    residual[x, y] -= min_residual
                    residual[y, x] += min_residual  # reverse edge
                total_flow += min_residual
                continue

            # Explore neighbors respecting level graph and positive residual
            neighbors = cp.where((residual[u, :] > 0) & (self.level[u] + 1 == self.level))[0]
            for v in neighbors:
                stack.append((v, cp.minimum(path_flow, residual[u, v]), path))

            # Explore neighbors starting from ptr[u]
            # neighbors = cp.where((residual[u, :] > 0) & (level[u] + 1 == level))[0]
            # idx = ptr[u]
            # while idx < len(neighbors):
            #     v = int(neighbors[idx])
            #     ptr[u] = idx + 1  # increment pointer
            #     idx += 1
            #     if residual[u, v] > 0:
            #         stack.append((v, cp.minimum(path_flow, residual[u, v])))
            #         paths.append(path)
            #         break  # move deeper along this neighbor

        return total_flow

    def max_flow(self, s, t):
        # Corner case
        if s == t:
            return -1

        total_flow = 0.0
        while True:
            # Step 1: Build level graph (GPU BFS)
            if not self.construct_levels(s, t):
                break  # no more augmenting paths

            # Step 2: Send blocking flows along DFS paths
            residual = self.adj.copy()
            flow = cp.zeros_like(self.adj)

            flow_increment = self.send_flow(residual, flow, s, t)
            if flow_increment == 0:
                break  # all blocking flows sent
            total_flow += flow_increment

        return total_flow

        # V = self.adj.shape[0]
        # residual = self.adj.copy()
        # flow = cp.zeros_like(self.adj)

        # # Initialize result
        # total_flow = 0

        # frontier = cp.zeros(V, dtype=cp.bool_)
        # frontier[s] = True
        # excess = cp.zeros(V, dtype=cp.float32)
        # excess[s] = cp.inf  # Source has infinite supply

        # while frontier.any():
        #     # eligible edges: frontier->next layer edges
        #     eligible = (frontier[:, None]) & (residual > 0) & (self.level[:, None] + 1 == self.level[None, :])

        #     print("Residual flow:")
        #     print(residual)
        #     print("Excess flow:")
        #     print(excess)
        #     print("---")

        #     # Compute flow to push along eligible edges
        #     push_amount = cp.minimum(residual, excess[:, None])
        #     push_amount = push_amount * eligible.astype(push_amount.dtype)

        #     # Update residual and flow
        #     residual -= push_amount
        #     flow += push_amount

        #     # Update excess for next layer
        #     excess = cp.sum(push_amount, axis=0)
        #     frontier = excess > 0

        # total_flow += flow[s, :].sum()  # Flow pushed from source

        # return flow, total_flow
