import time
import cupy as cp

# from test import simple_graphs
from test import generate_graphs
from test import stats
from src import dinics
from src import pushrelabel

WARMUP = 3
REPREAT = 30
TESTS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

gpu_stats = []
cpu_stats = []

# for g in simple_graphs.simple_graphs():
for T in TESTS:
    g = generate_graphs.generate_random_graph(T)

    PR = pushrelabel.PushRelabelGPU(g)

    for _ in range(WARMUP):
        PR.compute_max_flow(0, g.vcount() - 1, 1000)

    cp.cuda.Device().synchronize

    gpu_times = []
    cpu_times = []

    for _ in range(REPREAT):
        gpu_start = time.perf_counter()
        gpu_out = PR.compute_max_flow(0, g.vcount() - 1, 10)
        gpu_end = time.perf_counter()
        cp.cuda.Device().synchronize

        cpu_start = time.perf_counter()
        cpu_out = g.maxflow(0, g.vcount() - 1, capacity=g.es["capacity"]).value
        cpu_end = time.perf_counter()

        gpu_times.append(gpu_end - gpu_start)
        cpu_times.append(cpu_end - cpu_start)

    gpu_stats.append((g.vcount(), stats.runtime_stats(gpu_times)))
    cpu_stats.append((g.vcount(), stats.runtime_stats(cpu_times)))

    print(f"CuPy Output: {gpu_out}")
    print(f"iGraph Output: {cpu_out}")
    print()

print()
print("Num Nodes: med(ms), low(ms), high(ms), std_dev(ms)")
print("--------------------------------------------------")
print("CuPy Output:")
for graph_size, entry in gpu_stats:
    print(f"{graph_size} Nodes:", entry)
print()
print("iGraph Output:")
for graph_size, entry in cpu_stats:
    print(f"{graph_size} Nodes:", entry)
