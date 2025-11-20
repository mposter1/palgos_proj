import igraph as ig
import matplotlib.pyplot as plt

# Construct a graph with 5 vertices
g = ig.Graph(
    6,
    [(3, 2), (3, 4), (2, 1), (4,1), (4, 5), (1, 0), (5, 0)],
    directed=True
)
g.es["capacity"] = [7, 8, 1, 2, 3, 4, 5]

flow = g.maxflow(3, 0, capacity=g.es["capacity"])

print(g)
print(g.es["capacity"])

print("Max flow:", flow.value)
print("Edge assignments:", flow.flow)

# Output:
# Max flow: 6.0
# Edge assignments [1.0, 5.0, 1.0, 2.0, 3.0, 3.0, 3.0]

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout="circle",
    vertex_label=range(g.vcount()),
    vertex_color="lightblue",
    edge_label=[f"{w:.2g}" for w in flow.flow],
)

fig.savefig("test.png")
