import numpy as np


def reduce_scatter_owner(grads, world_size):
    reduced = np.sum(np.asarray(grads, dtype=np.float64), axis=0)
    shard_size = len(reduced) // world_size
    return [
        reduced[r * shard_size:(r + 1) * shard_size].tolist()
        for r in range(world_size)
    ]
