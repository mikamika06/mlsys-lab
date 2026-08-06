import numpy as np


def unflatten_all_gathered(shards: list[np.ndarray], shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    flat_list = []
    for shard in shards:
        arr = np.asarray(shard)
        for val in arr.flat:
            flat_list.append(val)
    
    total = 0
    for shape in shapes:
        prod = 1
        for dim in shape:
            prod = prod * dim
        total = total + prod
    
    flat = np.array(flat_list[:total], dtype=flat_list[0].dtype if flat_list else np.float64) if flat_list else np.array([], dtype=np.float64)

    params = []
    offset = 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size = size * dim
        
        chunk = flat[offset:offset + size]
        params.append(chunk.reshape(shape))
        offset += size
    return params
