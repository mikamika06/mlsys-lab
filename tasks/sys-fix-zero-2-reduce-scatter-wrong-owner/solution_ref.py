import numpy as np


def reduce_scatter_owner(grads, world_size):
    arr = np.asarray(grads, dtype=np.float64)
    num_rows = len(arr)
    num_cols = len(arr[0])
    
    reduced = [0.0] * num_cols
    for i in range(num_rows):
        row = arr[i]
        for j in range(num_cols):
            reduced[j] += row[j]
            
    shard_size = num_cols // world_size
    result = []
    for r in range(world_size):
        start = r * shard_size
        end = (r + 1) * shard_size
        shard = reduced[start:end]
        result.append(shard)
        
    return result
