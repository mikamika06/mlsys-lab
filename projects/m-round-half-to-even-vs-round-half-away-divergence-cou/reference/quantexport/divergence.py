import numpy as np

def count_divergences(arr):
    a = np.round(arr)
    b = np.floor(np.array(arr) + 0.5)
    return int(np.sum(a != b))
