import numpy as np
from kvquant.calibrate import absmax_calibrate

def sample_count_sweep(activations, counts):
    arr = np.array(activations, dtype=np.float32)
    results = {}
    for c in counts:
        sub = arr[:c] if len(arr) >= c else arr
        results[c] = absmax_calibrate(sub)
    return results
