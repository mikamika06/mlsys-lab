import numpy as np


def row_parallel_linear(x_shards, w_shards, bias):
    partials = []
    for x, w in zip(x_shards, w_shards):
        partials.append(
            np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
        )
    return np.sum(partials, axis=0) + np.asarray(bias, dtype=np.float64)
