import numpy as np


def simulate_fsdp_shards(params, world_size=2):
    shards = []
    for rank in range(world_size):
        rank_shards = {}
        for name, arr in sorted(params.items()):
            flat = arr.ravel()
            chunk_size = int(np.ceil(len(flat) / world_size))
            start = rank * chunk_size
            end = min(len(flat), (rank + 1) * chunk_size)
            rank_shards[name] = flat[start:end].copy()
        shards.append(rank_shards)
    return shards
