import numpy as np


def write_kv_cache(cache_k, cache_v, new_k, new_v, position):
    updated_k = np.array(cache_k, copy=True)
    updated_v = np.array(cache_v, copy=True)
    updated_k[position, :] = new_k
    updated_v[position, :] = new_v
    return updated_k, updated_v
