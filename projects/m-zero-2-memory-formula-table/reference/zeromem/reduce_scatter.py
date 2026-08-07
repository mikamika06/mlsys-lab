import numpy as np

def toy_reduce_scatter(grads_list, world_size):
    stacked = np.array(grads_list, dtype=np.float64)
    reduced_sum = np.sum(stacked, axis=0)
    chunk_size = len(reduced_sum) // world_size
    chunks = []
    for r in range(world_size):
        start = r * chunk_size
        end = (r + 1) * chunk_size if r < world_size - 1 else len(reduced_sum)
        chunks.append(reduced_sum[start:end])
    return chunks
