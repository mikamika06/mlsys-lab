import numpy as np


def toy_reduce_scatter(gradients, world_size, rank):
    arr = np.array(gradients, dtype=np.float32)
    reduced = arr / world_size
    chunks = np.array_split(reduced, world_size)
    return chunks[rank].tolist()
