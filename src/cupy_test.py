import cupy as cp
# from cupyx.distributed import device

for device in range(cp.cuda.runtime.getDeviceCount()):
    print(f'cuda:{device}')

# data = cp.arange(1000000)  # Create sample array on default GPU

# Split data across devices
# chunks = [data[i::len(devices)] for i in range(len(devices))]
# for i, dev in enumerate(devices):
#     with dev:
#         dev_data = cp.asarray(chunks[i])  # Transfer chunk to target GPU

# with cp.cuda.Device(0):
#     x = cp.array([1, 2, 3, 4, 5])
# print(x.device)

# with cp.cuda.Device(1):
#     y = cp.array([1, 2, 3, 4, 5])
# print(y.device)
# with cp.cuda.Device

