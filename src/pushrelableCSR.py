import numpy as np
import cupy as cp
import cupy.sparse as cps
import scipy.sparse as sps

# ------------------------------------------------------------
# Kernels (Unchanged logic, just accepting CSR arrays)
# ------------------------------------------------------------

select_kernel_code = r'''
extern "C" __global__
void select_push_candidate(int V,
                           const int* __restrict__ indptr,
                           const int* __restrict__ indices,
                           const float* __restrict__ data,
                           const int* __restrict__ h,
                           const float* __restrict__ excess,
                           int* __restrict__ chosen_edge,
                           int s, int t)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    chosen_edge[u] = -1;
    if (u == s || u == t || excess[u] <= 0.0f) return;

    int start = indptr[u];
    int end = indptr[u+1];
    int best_edge = -1;

    // Iterate over CSR row
    for (int i = start; i < end; i++) {
        int v = indices[i];
        float cap = data[i];

        if (cap > 0.0f) {
            if (h[u] == h[v] + 1) {
                best_edge = i;
                break; 
            }
        }
    }
    chosen_edge[u] = best_edge;
}
'''

push_kernel_code = r'''
extern "C" __global__
void push_atomic_csr(int V,
                     const int* __restrict__ indptr,
                     const int* __restrict__ indices,
                     const int* __restrict__ rev_map,
                     const int* __restrict__ chosen_edge,
                     float* __restrict__ data,
                     float* __restrict__ excess)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    int edge_idx = chosen_edge[u];
    if (edge_idx < 0) return;

    int v = indices[edge_idx];
    int rev_idx = rev_map[edge_idx];

    float d = min(excess[u], data[edge_idx]);

    if (d <= 0.0f) return;

    // Atomic updates on the sparse data array
    atomicAdd(&data[edge_idx], -d);
    atomicAdd(&data[rev_idx],  d);

    atomicAdd(&excess[u], -d);
    atomicAdd(&excess[v],  d);
}
'''

relabel_kernel_code = r'''
extern "C" __global__
void relabel_csr(int V,
                 const int* __restrict__ indptr,
                 const int* __restrict__ indices,
                 const float* __restrict__ data,
                 int* __restrict__ h,
                 const float* __restrict__ excess,
                 const int* __restrict__ chosen_edge,
                 int s, int t)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    if (u == s || u == t || excess[u] <= 0.0f || chosen_edge[u] != -1) return;

    int start = indptr[u];
    int end = indptr[u+1];
    int min_height = 2147483640;
    bool possible = false;

    for (int i = start; i < end; i++) {
        if (data[i] > 0.0f) {
            int v = indices[i];
            if (h[v] < min_height) {
                min_height = h[v];
                possible = true;
            }
        }
    }

    if (possible) {
        h[u] = min_height + 1;
    }
}
'''

mod_select = cp.RawModule(code=select_kernel_code)
k_select = mod_select.get_function("select_push_candidate")

mod_push = cp.RawModule(code=push_kernel_code)
k_push = mod_push.get_function("push_atomic_csr")

mod_relabel = cp.RawModule(code=relabel_kernel_code)
k_relabel = mod_relabel.get_function("relabel_csr")


class PushRelabelCSR:
    def __init__(self, ig):
        """
        ig: igraph Graph object
        Uses cupy.scipy.sparse.csr_matrix for storage.
        """
        self.V = ig.vcount()
        
        # -------------------------------------------------------
        # 1. PREPROCESSING (CPU)
        # -------------------------------------------------------
        # We use Scipy first to handle the graph construction and 
        # sorting of indices, so we can calculate rev_map reliably.
        
        edges = ig.get_edgelist()
        caps = ig.es["capacity"]
        
        # Build a symmetric map: (u,v) -> capacity
        cap_map = {}
        for (u, v), c in zip(edges, caps):
            cap_map[(u, v)] = cap_map.get((u, v), 0) + c
            if (v, u) not in cap_map:
                cap_map[(v, u)] = 0.0
        
        # Flatten to COO lists
        src = []
        dst = []
        data = []
        for (u, v), c in cap_map.items():
            src.append(u)
            dst.append(v)
            data.append(c)
            
        # Create SCIPY CSR Matrix (CPU)
        # Scipy will automatically sort indices and sum duplicates
        cpu_csr = sps.csr_matrix((data, (src, dst)), shape=(self.V, self.V), dtype=np.float32)
        
        # -------------------------------------------------------
        # 2. BUILD REVERSE MAP (CPU)
        # -------------------------------------------------------
        # Now that Scipy has organized the memory layout, we need to know:
        # If edge (u->v) is at index 'k' in .data, where is (v->u)?
        
        # Create a lookup: (u, v) -> index in .data array
        edge_lookup = {}
        
        # cpu_csr.indptr and cpu_csr.indices tell us exactly where edges are stored
        for u in range(self.V):
            start = cpu_csr.indptr[u]
            end = cpu_csr.indptr[u+1]
            for idx in range(start, end):
                v = cpu_csr.indices[idx]
                edge_lookup[(u, v)] = idx
        
        # Build the reverse map array
        # rev_map[k] = index of the reverse edge for the edge at index k
        rev_map_cpu = np.zeros(cpu_csr.nnz, dtype=np.int32)
        
        for u in range(self.V):
            start = cpu_csr.indptr[u]
            end = cpu_csr.indptr[u+1]
            for idx in range(start, end):
                v = cpu_csr.indices[idx]
                # Find where (v, u) lives
                rev_idx = edge_lookup[(v, u)]
                rev_map_cpu[idx] = rev_idx

        # -------------------------------------------------------
        # 3. TRANSFER TO GPU (Cupy CSR)
        # -------------------------------------------------------
        
        # We can initialize Cupy CSR directly from Scipy CSR
        self.graph = cps.csr_matrix(cpu_csr)
        
        # Store raw pointers for the kernel (pointers to GPU memory)
        # Cupy sparse matrices expose .data, .indices, .indptr as cupy arrays
        self.rev_map = cp.array(rev_map_cpu, dtype=cp.int32)
        
        self.h = cp.zeros(self.V, dtype=cp.int32)
        self.excess = cp.zeros(self.V, dtype=cp.float32)
        self.chosen_edge = cp.full(self.V, -1, dtype=cp.int32)

    def initialize_preflow(self, s):
        self.h[s] = self.V
        
        # Get source range from the GPU arrays
        start = int(self.graph.indptr[s])
        end = int(self.graph.indptr[s+1])
        
        # We need to modify self.graph.data.
        # While we can do this via kernel, for initialization simple slicing is fine.
        
        # Extract source edges to CPU for logic (or use specific kernel)
        # For simplicity in Python:
        s_indices = self.graph.indices[start:end].get()
        s_data = self.graph.data[start:end].get()
        s_rev_indices = self.rev_map[start:end].get()
        
        total_pushed = 0.0
        
        # We have to update the main GPU data array
        # To avoid slow individual cp updates, we compute updates on CPU then push back
        
        # Copy entire data array to CPU temp (for easy scattered updates)
        cpu_data = self.graph.data.get()
        cpu_excess = np.zeros(self.V, dtype=np.float32)
        
        for i in range(len(s_indices)):
            cap = s_data[i]
            if cap > 0:
                v = s_indices[i]
                rev_idx = s_rev_indices[i]
                
                # Forward: s->v (index start+i)
                cpu_data[start + i] = 0
                
                # Reverse: v->s (index rev_idx)
                cpu_data[rev_idx] += cap
                
                cpu_excess[v] += cap
                cpu_excess[s] -= cap
                
        # Push back modified arrays
        self.graph.data = cp.array(cpu_data, dtype=cp.float32)
        self.excess = cp.array(cpu_excess, dtype=cp.float32)

    def step(self, s, t):
        threads = 128
        blocks = (self.V + threads - 1) // threads
        
        # Access underlying Cupy arrays
        # csr_matrix.indptr -> row offsets
        # csr_matrix.indices -> col indices
        # csr_matrix.data -> values (residual capacity)
        
        # 1. Select
        k_select((blocks,), (threads,), (
            self.V, 
            self.graph.indptr, 
            self.graph.indices, 
            self.graph.data, 
            self.h, 
            self.excess, 
            self.chosen_edge, 
            s, t
        ))
        
        # Check termination
        # (Optimization: checking excess on GPU avoids device->host sync every step, 
        # but for this structure we return boolean to control the loop)
        has_excess = cp.any((self.excess > 0) & (cp.arange(self.V) != s) & (cp.arange(self.V) != t))
        if not has_excess:
            return False

        # 2. Push
        k_push((blocks,), (threads,), (
            self.V, 
            self.graph.indptr, 
            self.graph.indices, 
            self.rev_map, 
            self.chosen_edge, 
            self.graph.data, 
            self.excess
        ))

        # 3. Relabel
        k_relabel((blocks,), (threads,), (
            self.V, 
            self.graph.indptr, 
            self.graph.indices, 
            self.graph.data, 
            self.h, 
            self.excess, 
            self.chosen_edge, 
            s, t
        ))

        return True

    def compute_max_flow(self, s, t, max_iter=1_000_000):
        self.initialize_preflow(s)

        for i in range(max_iter):
            active = self.step(s, t)

            if not active:
                break

        return float(self.excess[t].get())
