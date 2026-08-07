import numpy as np

def check_memory_stability(memory_samples: list) -> bool:
    if not memory_samples:
        return False
    arr = np.array(memory_samples)
    drift = arr[-1] - arr[0]
    return drift < 5.0 and np.std(arr) < 2.0
