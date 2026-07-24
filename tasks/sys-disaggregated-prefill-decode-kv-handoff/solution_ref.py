import numpy as np

def assemble_cache(chunks):
    # assume chunks are in order of increasing start_pos
    L = len(chunks[0][1])
    keys_by_layer = [[] for _ in range(L)]
    vals_by_layer = [[] for _ in range(L)]
    for _, layer_kv in chunks:
        for i, (k, v) in enumerate(layer_kv):
            keys_by_layer[i].append(k)
            vals_by_layer[i].append(v)
    return [(np.concatenate(keys, axis=2), np.concatenate(vals, axis=2))
            for keys, vals in zip(keys_by_layer, vals_by_layer)]
