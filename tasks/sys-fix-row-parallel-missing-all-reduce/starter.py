import numpy as np


def row_parallel_linear(x_shards, w_shards, bias):
    # TODO: missing all-reduce. This returns only worker 0's partial output.
    partial = np.asarray(x_shards[0], dtype=np.float64) @ np.asarray(
        w_shards[0], dtype=np.float64
    )
    return partial + np.asarray(bias, dtype=np.float64)
