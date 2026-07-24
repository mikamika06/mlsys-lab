import numpy as np


def flatten_pad_shard(params, world_size):
    flat = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])
    total = flat.size
    remainder = total % world_size
    pad = 0 if remainder == 0 else (world_size - remainder)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float64)])
    shard_size = flat.size // world_size
    return [flat[i * shard_size:(i + 1) * shard_size] for i in range(world_size)]
