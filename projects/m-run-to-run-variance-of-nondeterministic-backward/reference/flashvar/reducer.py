import numpy as np


def deterministic_backward(grads):
    arr = np.array(grads, dtype=np.float64)
    sorted_arr = np.sort(arr, axis=0)
    accumulated = np.sum(sorted_arr, axis=0)
    return accumulated.tolist()
