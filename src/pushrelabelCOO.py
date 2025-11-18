import numpy as np
import cupy as cp
import cupy.sparse as cps
import scipy.sparse as sps

# ------------------------------------------------------------
# Device Function: Binary Search
# ------------------------------------------------------------
# Since COO doesn't have a row pointer, we must search for the
# starting position of a node's edges in the huge row_indices array.

device_func = r'''
extern "C" __device__
int lower_bound(const int* arr, int val, int n) {
    int l = 0;
    int r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] < val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}
'''

# ------------------------------------------------------------
# Kernels: COO Logic
# ------------------------------------------------------------

select_kernel_code = device_func + r'''
extern "C" __global__
void select_push_candidate_coo(int V, int E,
                               const int* __restrict__ row_ind,
                               const int* __restrict__ col_ind,
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

    // --- THE COO BOTTLENECK ---
    // We have to binary search the row_ind array to find where u starts.
    int start = lower_bound(row_ind, u, E);
    
    int best_edge = -1;

    // Iterate while the row index is still u
    for (int i = start; i < E; i++) {
        if (row_ind[i] != u) break; // We moved past node u's edges

        int v = col_ind[i];
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
void push_atomic_coo(int V,
                     const int* __restrict__ col_ind,
                     const int* __restrict__ rev_map,
                     const int* __restrict__ chosen_edge,
                     float* __restrict__ data,
                     float* __restrict__ excess)
{
    // Push logic is identical to CSR because we already resolved 
    // the "chosen_edge" index in the previous kernel.
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    int edge_idx = chosen_edge[u];
    if (edge_idx < 0) return;

    int v = col_ind[edge_idx];
    int rev_idx = rev_map[edge_idx];

    float d = min(excess[u], data[edge_idx]);

    if (d <= 0.0f) return;

    atomicAdd(&data[edge_idx], -d);
    atomicAdd(&data[rev_idx],  d);
    atomicAdd(&excess[u], -d);
    atomicAdd(&excess[v],  d);
}
'''

relabel_kernel_code = device_func + r'''
extern "C" __global__
void relabel_coo(int V, int E,
                 const int* __restrict__ row_ind,
                 const int* __restrict__ col_ind,
                 const float* __restrict__ data,
                 int* __restrict__ h,
                 const float* __restrict__ excess,
                 const int* __restrict__ chosen_edge,
                 int s, int t)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    if (u == s || u == t || excess[u] <= 0.0f || chosen_edge[u] != -1) return;

    // --- THE COO BOTTLENECK ---
    int start = lower_bound(row_ind, u, E);
    
    int min_height = 2147483640;
    bool possible = false;

    for (int i = start; i < E; i++) {
        if (row_ind[i] != u) break;

        if (data[i] > 0.0f) {
            int v = col_ind[i];
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
k_select = mod_select.get_function("select_push_candidate_coo")

mod_push = cp.RawModule(code=push_kernel_code)
k_push = mod_push.get_function("push_atomic_coo")

mod_relabel = cp.RawModule(code=relabel_kernel_code)
k_relabel = mod_relabel.get_function("relabel_coo")


class PushRelabelCOO:
    def __init__(self, ig):
        self.V = ig.vcount()
        
        # 1. PREPROCESSING (CPU)
        edges = ig.get_edgelist()
        caps = ig.es["capacity"]
        
        # Symmetric map construction
        cap_map = {}
        for (u, v), c in zip(edges, caps):
            cap_map[(u, v)] = cap_map.get((u, v), 0) + c
            if (v, u) not in cap_map:
                cap_map[(v, u)] = 0.0
        
        # Extract to simple Python lists (guaranteed 1-D)
        src_list = []
        dst_list = []
        data_list = []
        for (u, v), c in cap_map.items():
            src_list.append(u)
            dst_list.append(v)
            data_list.append(c)
            
        # Convert to numpy arrays immediately to control shape/type
        # .flatten() ensures 1-D. .astype() ensures CUDA compatibility.
        src_np = np.array(src_list).flatten().astype(np.int32)
        dst_np = np.array(dst_list).flatten().astype(np.int32)
        data_np = np.array(data_list).flatten().astype(np.float32)
        
        # 2. SORTING (Crucial for Binary Search in Kernel)
        # We use lexsort to get indices that sort by Row first, then Col
        order = np.lexsort((dst_np, src_np))
        
        # Apply sort order
        self.row = src_np[order]
        self.col = dst_np[order]
        self.data = data_np[order]
        
        # Safety Check: Assertions to guarantee constraints before GPU transfer
        assert self.row.ndim == 1, "Row indices must be 1-D"
        assert self.col.ndim == 1, "Col indices must be 1-D"
        assert self.data.ndim == 1, "Data must be 1-D"
        assert len(self.row) == len(self.col) == len(self.data), "COO arrays length mismatch"

        self.E = len(self.data)
        
        # 3. BUILD REVERSE MAP (CPU)
        # Create lookup: (u, v) -> index
        # Since inputs are flattened, this logic remains simple
        edge_lookup = {}
        for idx in range(self.E):
            u = self.row[idx]
            v = self.col[idx]
            edge_lookup[(u, v)] = idx
            
        rev_map_cpu = np.zeros(self.E, dtype=np.int32)
        for idx in range(self.E):
            u = self.row[idx]
            v = self.col[idx]
            rev_map_cpu[idx] = edge_lookup[(v, u)]

        # 4. TRANSFER TO GPU
        # Convert to CuPy arrays FIRST.
        # This ensures they are strictly 1-D and residing on GPU memory.
        cp_row = cp.array(self.row)
        cp_col = cp.array(self.col)
        cp_data = cp.array(self.data)

        self.graph = cps.coo_matrix((cp_data, (cp_row, cp_col)), 
                                    shape=(self.V, self.V))
        
        # We still store raw arrays for our kernels (RawModules need pointers)
        self.graph.row = cp_row
        self.graph.col = cp_col
        self.graph.data = cp_data
        
        self.rev_map = cp.array(rev_map_cpu, dtype=cp.int32)
        
        self.h = cp.zeros(self.V, dtype=cp.int32)
        self.excess = cp.zeros(self.V, dtype=cp.float32)
        self.chosen_edge = cp.full(self.V, -1, dtype=cp.int32)

    def initialize_preflow(self, s):
        self.h[s] = self.V
        
        # --- CPU Side Init for simplicity ---
        # We use binary search (searchsorted) on CPU to find source edges
        start = np.searchsorted(self.row, s, side='left')
        end = np.searchsorted(self.row, s, side='right')
        
        cpu_data = self.graph.data.get()
        cpu_excess = np.zeros(self.V, dtype=np.float32)
        rev_map_cpu = self.rev_map.get()
        
        for i in range(start, end):
            cap = cpu_data[i]
            if cap > 0:
                v = self.col[i]
                rev_idx = rev_map_cpu[i]
                
                cpu_data[i] = 0
                cpu_data[rev_idx] += cap
                cpu_excess[v] += cap
                cpu_excess[s] -= cap
                
        self.graph.data = cp.array(cpu_data, dtype=cp.float32)
        self.excess = cp.array(cpu_excess, dtype=cp.float32)

    def step(self, s, t):
        threads = 128
        blocks = (self.V + threads - 1) // threads
        
        # 1. Select
        # Notice we pass self.graph.row (size E) instead of indptr (size V)
        k_select((blocks,), (threads,), (
            self.V, self.E,
            self.graph.row, 
            self.graph.col, 
            self.graph.data, 
            self.h, 
            self.excess, 
            self.chosen_edge, 
            s, t
        ))
        
        has_excess = cp.any((self.excess > 0) & (cp.arange(self.V) != s) & (cp.arange(self.V) != t))
        if not has_excess:
            return False

        # 2. Push
        k_push((blocks,), (threads,), (
            self.V, 
            self.graph.col, 
            self.rev_map, 
            self.chosen_edge, 
            self.graph.data, 
            self.excess
        ))

        # 3. Relabel
        k_relabel((blocks,), (threads,), (
            self.V, self.E,
            self.graph.row, 
            self.graph.col, 
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
