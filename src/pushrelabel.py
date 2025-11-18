import numpy as np
import cupy as cp

# ------------------------------------------------------------
# Tiny kernel: apply pushes atomically
# ------------------------------------------------------------

push_kernel_code = r'''
extern "C" __global__
void push_atomic(int V,
                 const int* __restrict__ u_idx,
                 const int* __restrict__ v_idx,
                 const float* __restrict__ d_push,
                 float* __restrict__ rmat,
                 float* __restrict__ excess)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= V) return;

    int u = u_idx[k];
    int v = v_idx[k];

    if (v < 0) return;   // No admissible neighbor

    float d = d_push[k]; // push amount
    if (d <= 0.0f) return;

    int uv = u*V + v;
    int vu = v*V + u;

    // forward residual -= d
    atomicAdd(&rmat[uv], -d);
    // reverse residual += d
    atomicAdd(&rmat[vu],  d);

    // update excess
    atomicAdd(&excess[u], -d);
    atomicAdd(&excess[v],  d);
}
'''

push_mod = cp.RawModule(code=push_kernel_code)
push_kernel = push_mod.get_function("push_atomic")


class PushRelabelGPU:
    """
    Sinlge GPU implementation of the push-relabel algorithm for max flow
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

    def initialize_preflow(self, s):
        self.h[s] = self.V  # source height = V

        for i in range(self.V):
            if self.adj[s][i] > 0:
                self.rmat[s][i] = self.adj[s][i] + self.adj[i][s]
                self.excess[i] = self.adj[s][i]
                self.excess_total += self.excess[i]

    def step(self, s, t):
        V = self.V

        # we don't push from sink or source
        active = (self.excess > 0) & (cp.arange(V) != s) & (cp.arange(V) != t)
        if not cp.any(active):
            return False

        # ---------------------------
        # (1) Compute push candidates
        # ---------------------------
        # residual > 0
        res_mask = (self.rmat > 0)

        # height difference h[u] == h[v] + 1
        h_expanded_u = self.h[:, None]         # shape (V,1)
        h_expanded_v = self.h[None, :]         # shape (1,V)
        admissible = (h_expanded_u == h_expanded_v + 1)

        # combine
        can_push = res_mask & admissible       # shape (V, V)

        # For vertices that are not active, disable pushing
        can_push &= active[:, None]

        # ---------------------------
        # (2) Select exactly one admissible neighbor (vectorized)
        # ---------------------------
        # Give each neighbor either its height or INF
        INF = cp.int32(2**30)
        masked_heights = cp.where(can_push, self.h[None, :], INF)

        # Select neighbor with minimum height for each u
        v_idx = cp.argmin(masked_heights, axis=1)    # shape (V,)

        # If a row has no admissible neighbors, its min-height = INF; fix that:
        v_idx = cp.where(masked_heights.min(axis=1) < INF, v_idx, -1)

        # ---------------------------
        # (3) Compute push amounts d[u] = min(excess[u], rmat[u,v])
        # ---------------------------
        u_idx = cp.arange(V, dtype=cp.int32)

        r_uv = self.rmat[u_idx, v_idx]  # residual capacity for chosen neighbor
        r_uv = cp.where(v_idx >= 0, r_uv, 0.0)

        d_push = cp.minimum(self.excess, r_uv)

        # ---------------------------
        # (4) Launch tiny atomic push kernel
        # ---------------------------
        threads = 128
        blocks = (V + threads - 1) // threads
        push_kernel((blocks,), (threads,),
                    (V,
                     u_idx.astype(cp.int32),
                     v_idx.astype(cp.int32),
                     d_push.astype(cp.float32),
                     self.rmat,
                     self.excess))

        # ---------------------------
        # (5) Relabel nodes with positive excess that couldn't push
        # ---------------------------
        # Those with v_idx == -1 can't push
        stuck = (self.excess > 0) & (v_idx == -1) & (cp.arange(V) != s) & (cp.arange(V) != t)

        if cp.any(stuck):
            # Compute new minimal height among neighbors with residual > 0
            neighbor_heights = cp.where(self.rmat > 0, self.h[None, :], INF)
            min_h = neighbor_heights.min(axis=1)
            self.h = cp.where(stuck, min_h + 1, self.h)

        return True

    def compute_max_flow(self, s, t, max_iter=1_000_000):
        self.initialize_preflow(s)

        for _ in range(max_iter):
            active = (self.excess > 0) & (cp.arange(self.V) != s) & (cp.arange(self.V) != t)
            if not cp.any(active):
                break
            self.step(s, t)

        # Max flow = sum of excess pushed into sink
        return float(self.excess[t].get())
