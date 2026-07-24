import numpy as np


def mixed_rank_sgmv(x, adapter_ids, adapters):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        a, b = adapters[int(adapter_ids[i])]
        out[i] = x[i] + b @ (a @ x[i])
    return out
