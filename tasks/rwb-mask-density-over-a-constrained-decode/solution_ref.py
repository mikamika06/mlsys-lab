import numpy as np


def mask_density_trace(vocab_size, trace, allowed):
    densities = []
    for state in trace:
        densities.append(
            (vocab_size - len(set(allowed[state]))) / vocab_size
        )
    arr = np.asarray(densities, dtype=np.float64)
    return arr, float(np.mean(arr))
