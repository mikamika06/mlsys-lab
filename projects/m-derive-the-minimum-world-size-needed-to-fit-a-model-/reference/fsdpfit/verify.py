import numpy as np


def verify_all_gathered(original, shards):
    world_size = len(shards)
    reconstructed = {}
    for name, arr in original.items():
        flat_shape = arr.shape
        flat_parts = [shards[r][name] for r in range(world_size)]
        combined = np.concatenate(flat_parts)
        reconstructed[name] = combined.reshape(flat_shape)
    for name, arr in original.items():
        if not np.allclose(arr, reconstructed[name]):
            return False
    return True
