import time
import numpy as np
import cupy as cp

from test import generate_graphs
from test import stats
from src.pushrelabelCOO import PushRelabelCOO

REPREAT = 100
TESTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

gpu_stats = []

for T in TESTS:
    g = generate_graphs.generate_random_graph(T, seed=69420)

    gpu_times = []
    results = 0

    for _ in range(REPREAT):
        PR = PushRelabelCOO(g)

        cp.cuda.Device().synchronize
        gpu_start = time.perf_counter()
        gpu_out = PR.compute_max_flow(0, g.vcount() - 1, 1000)
        gpu_end = time.perf_counter()
        cp.cuda.Device().synchronize

        gpu_times.append(gpu_end - gpu_start)
        results += gpu_out

    gpu_stats.append((g.vcount(), stats.runtime_stats(gpu_times)))

    print(f"COO Output for {g.vcount()} nodes: {results / REPREAT}")

print()
print("Sparse COO Results:")
print()
print("Num Nodes, med(ms), low(ms), high(ms), std_dev(ms)")
print("--------------------------------------------------")
for graph_size, entry in gpu_stats:
    print(f"{graph_size},", entry)
print()

