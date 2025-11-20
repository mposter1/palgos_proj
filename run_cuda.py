import time
import cupy as cp

# from test import simple_graphs
from test import generate_graphs
from test import stats
from src.pushrelabel import PushRelabelGPU
from src.pushrelableCSR import PushRelabelCSR
from src.pushrelabelCOO import PushRelabelCOO

REPREAT = 10
# TESTS = [10
TESTS = [2000, 4000, 6000] 

cpu_stats = []
dense_stats = []
csr_stats = []
coo_stats = []

for T in TESTS:
    cpu_times = []
    dense_times = []
    csr_times = []
    coo_times = []

    cpu_out = 0
    dense_out = 0
    csr_out = 0
    coo_out = 0

    for _ in range(REPREAT):
        g = generate_graphs.generate_random_graph(T, seed=69420)

        print(f"{T} nodes, iteration {_}")
        print()

        dense = PushRelabelGPU(g)
        sparse_csr = PushRelabelCSR(g)
        sparse_coo = PushRelabelCOO(g)

        cpu_start = time.perf_counter()
        cpu_out += g.maxflow(0, g.vcount() - 1, capacity=g.es["capacity"]).value
        cpu_end = time.perf_counter()

        cpu_times.append(cpu_end - cpu_start)

        cp.cuda.Device().synchronize

        dense_start = time.perf_counter()
        dense_out += dense.compute_max_flow(0, g.vcount() - 1, 1000)
        dense_end = time.perf_counter()

        cp.cuda.Device().synchronize
        dense_times.append(dense_end - dense_start)

        cp.cuda.Device().synchronize

        csr_start = time.perf_counter()
        csr_out += sparse_csr.compute_max_flow(0, g.vcount() - 1, 1000)
        csr_end = time.perf_counter()

        cp.cuda.Device().synchronize
        csr_times.append(csr_end - csr_start)

        cp.cuda.Device().synchronize

        coo_start = time.perf_counter()
        coo_out += sparse_coo.compute_max_flow(0, g.vcount() - 1, 1000)
        coo_end = time.perf_counter()

        cp.cuda.Device().synchronize
        coo_times.append(coo_end - coo_start)

    print(f"iGraph Output: {cpu_out / REPREAT}")
    print(f"Dense Output: {dense_out / REPREAT}")
    print(f"CSR Output: {csr_out / REPREAT}")
    print(f"COO Output: {coo_out / REPREAT}")
    print("---")

    cpu_stats.append((g.vcount(), stats.runtime_stats(cpu_times)))
    dense_stats.append((g.vcount(), stats.runtime_stats(dense_times)))
    csr_stats.append((g.vcount(), stats.runtime_stats(csr_times)))
    coo_stats.append((g.vcount(), stats.runtime_stats(coo_times)))

print()
print("Num Nodes: med(ms), low(ms), high(ms), std_dev(ms)")
print("--------------------------------------------------")
print("iGraph Output:")
for graph_size, entry in cpu_stats:
    print(f"{graph_size} Nodes:", entry)
print()
print("Dense CuPy Output:")
for graph_size, entry in dense_stats:
    print(f"{graph_size} Nodes:", entry)
print()
print("Sparse COO Output:")
for graph_size, entry in coo_stats:
    print(f"{graph_size} Nodes:", entry)
print()
print("Sparse CSR Output:")
for graph_size, entry in csr_stats:
    print(f"{graph_size} Nodes:", entry)
print()
