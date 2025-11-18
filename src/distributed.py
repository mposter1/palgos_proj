import numpy as np
import cupy as cp


class PushRelabelDistr:
    """
    Multi-GPU implementation of the push-relabel algorithm for max flow
    """

    def __init__(self, ig):
        """
        ig: igraph Graph object with capacities stored in es["capacity"]
        """
        self.V = ig.vcount()

        # Create dense adjacency matrix from igraph on the CPU
        self.cpu_adj = np.array(ig.get_adjacency(attribute="capacity").data, dtype=np.float32)

        self.adj = cp.asarray(self.cpu_adj)
        self.rmat = cp.asarray(self.cpu_adj.copy())  # residual capacity
        self.flow = cp.zeros_like(self.rmat)         # not used in push–relabel core
        self.h = cp.zeros(self.V, dtype=cp.int32)
        self.excess = cp.zeros(self.V, dtype=cp.float32)
        self.excess_total = 0



