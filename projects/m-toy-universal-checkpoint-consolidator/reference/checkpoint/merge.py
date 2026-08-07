import numpy as np

def merge_tp_shards(shards, axis_map):
    if not shards:
        return {}
    keys = shards[0].keys()
    out = {}
    for k in keys:
        axis = axis_map.get(k, None)
        if axis is None:
            out[k] = shards[0][k]
        else:
            out[k] = np.concatenate([s[k] for s in shards], axis=axis)
    return out
