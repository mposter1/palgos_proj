import cupy as cp
# import time

# # Print some info
# print("Starting GPU stress test with CuPy...")
# cp.cuda.runtime.setDevice(0)
# print("Using device:", cp.cuda.runtime.getDevice())

# # Create large random matrices (these will live on the GPU)
# N = 8192  # You can increase to 10000+ if you have enough VRAM
# a = cp.random.rand(N, N, dtype=cp.float32)
# b = cp.random.rand(N, N, dtype=cp.float32)

# # Warm up (JIT compilation)
# print("Warming up...")
# cp.dot(a[:1024, :1024], b[:1024, :1024])
# cp.cuda.Device(0).synchronize()

# # Main computation
# print("Starting heavy computation...")
# start = time.time()
# c = cp.dot(a, b)
# cp.cuda.Device(0).synchronize()
# end = time.time()

# print(f"Computation finished in {end - start:.2f} seconds.")
# print("Result sum:", float(c.sum()))

with cp.cuda.Device(0):
    x = cp.array([1, 2, 3, 4, 5])
print(x.device)

# with cp.cuda.Device(1):
#     y = cp.array([1, 2, 3, 4, 5])
# print(y.device)
