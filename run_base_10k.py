import time
import numpy as np
import cupy as cp

# from test import simple_graphs
from test import generate_graphs
from test import stats
from src import dinics
from src import pushrelabel

REPREAT = 100
TESTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

cpu_stats = []

for T in TESTS:
    g = generate_graphs.generate_random_graph(T, seed=69420)

    cpu_times = []
    results = []

    for _ in range(REPREAT):
        cpu_start = time.perf_counter()
        cpu_out = g.maxflow(0, g.vcount() - 1, capacity=g.es["capacity"]).value
        cpu_end = time.perf_counter()

        cpu_times.append(cpu_end - cpu_start)
        results.append(cpu_out)

    cpu_stats.append((g.vcount(), stats.runtime_stats(cpu_times)))

    print(f"iGRaph Output for {g.vcount()} nodes: {np.median(np.array(results))}")

print()
print("iGraph Results:")
print()
print("Num Nodes, med(ms), low(ms), high(ms), std_dev(ms)")
print("--------------------------------------------------")
for graph_size, entry in cpu_stats:
    print(f"{graph_size},", entry)
print()
