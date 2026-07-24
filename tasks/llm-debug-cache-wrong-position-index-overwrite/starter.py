import numpy as np


def write_kv_cache(cache_k, cache_v, new_k, new_v, position):
    # TODO: fix the off-by-one cache write. This writes to the previous token
    # position, overwriting an existing KV entry during decode.
    updated_k = np.array(cache_k, copy=True)
    updated_v = np.array(cache_v, copy=True)

    wrong_position = max(0, position - 1)
    updated_k[wrong_position, :] = new_k
    updated_v[wrong_position, :] = new_v

    return updated_k, updated_v
