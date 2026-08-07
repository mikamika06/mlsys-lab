import numpy as np


def simulate_reduce_scatter(gradients, world_size):
    flat = np.concatenate([g.ravel() for g in gradients])
    total_elements = flat.size
    remainder = total_elements % world_size
    if remainder != 0:
        pad_size = world_size - remainder
        flat = np.pad(flat, (0, pad_size), mode='constant')

    chunk_size = flat.size // world_size
    chunks = np.array_split(flat, world_size)

    reduced_shards = []
    for rank in range(world_size):
        shard_sum = np.zeros(chunk_size, dtype=np.float32)
        for r in range(world_size):
            shift_idx = (rank + r) % world_size
            shard_sum += chunks[shift_idx]
        reduced_shards.append(shard_sum)
    return reduced_shards
