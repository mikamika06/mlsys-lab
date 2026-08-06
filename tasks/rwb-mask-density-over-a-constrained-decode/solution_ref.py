import numpy as np


def mask_density_trace(vocab_size, trace, allowed):
    densities = []
    total = 0.0
    count = 0
    for state in trace:
        s = allowed[state]
        seen = {}
        unique_len = 0
        for item in s:
            if item not in seen:
                seen[item] = True
                unique_len += 1
        val = (vocab_size - unique_len) / vocab_size
        densities.append(val)
        total += val
        count += 1
    arr = np.asarray(densities, dtype=np.float64)
    mean_val = total / count if count > 0 else 0.0
    return arr, float(mean_val)
