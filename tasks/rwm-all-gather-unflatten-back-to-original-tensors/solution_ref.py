import numpy as np


def unflatten_all_gathered(shards: list[np.ndarray], shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    gathered = np.concatenate([np.asarray(shard).reshape(-1) for shard in shards])
    total = sum(int(np.prod(shape)) for shape in shapes)
    flat = gathered[:total]

    params = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        params.append(flat[offset:offset + size].reshape(shape))
        offset += size
    return params
