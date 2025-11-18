import numpy as np
import cupy as cp


class PushRelabelDistr:
    """
    Multi-GPU implementation of the push-relabel algorithm for max flow
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
        self.gpu_data = {device:[] for device in range(cp.cuda.runtime.getDeviceCount())}

    def allocate_gpu_adj(self):
        """
        Partition the adjacency matrix for the input graph across all the GPUs.
        May not have even partitions. I.e. if we have 10 nodes and 3 GPUs, then
        we have to split the 10x10 adj matrix into 3 partitions. In that case,
        it will be split into 4x10, 4x10, and 2x10
        """

        num_gpu = cp.cuda.runtime.getDeviceCount()

        partV = np.ceil(self.V / num_gpu).astype(np.int_)

        print(f"Partition graph across {num_gpu} devices")
        print('Original adjacency matrix:')
        print(self.adj)
        print()

        for device in range(num_gpu):
            # Get a 2x2 subarray
            # sub_matrix = matrix[0:2, 1:3]  
            start = device * partV
            end = start + partV

            partition = np.asarray(self.adj[start:end, 0:self.V-1])

            with cp.cuda.Device(device):
                self.gpu_data[device].append(
                        (f"gpu_{device}_adj",cp.asarray(partition)))

            for device in self.gpu_data.keys():
                print(f'Device cuda:{device} gets:')
                print(self.gpu_data[device])



