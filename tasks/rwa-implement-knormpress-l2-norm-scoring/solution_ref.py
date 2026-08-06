import math
import numpy as np

def knormpress(data: dict[int, np.ndarray], budget: int) -> list[int]:
    norms = {}
    for k, v in data.items():
        acc = 0.0
        for i in range(v.shape[0]):
            val = v[i]
            acc += val * val
        norms[k] = math.sqrt(acc)
    
    keys = list(norms.keys())
    n_keys = len(keys)
    for i in range(n_keys):
        for j in range(0, n_keys - i - 1):
            k1 = keys[j]
            k2 = keys[j + 1]
            if norms[k1] < norms[k2] or (norms[k1] == norms[k2] and k1 < k2):
                keys[j], keys[j + 1] = keys[j + 1], keys[j]
    
    top_keys = []
    limit = budget if budget < n_keys else n_keys
    for i in range(limit):
        top_keys.append(keys[i])
    
    n_top = len(top_keys)
    for i in range(n_top):
        for j in range(0, n_top - i - 1):
            if top_keys[j] > top_keys[j + 1]:
                top_keys[j], top_keys[j + 1] = top_keys[j + 1], top_keys[j]
                
    return top_keys
