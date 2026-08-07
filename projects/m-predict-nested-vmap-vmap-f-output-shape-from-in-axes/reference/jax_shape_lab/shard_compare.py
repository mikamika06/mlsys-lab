import numpy as np


def simulate_shard_vs_pmap(arr, axis_name):
    return np.sum(arr, axis=0, keepdims=True)
