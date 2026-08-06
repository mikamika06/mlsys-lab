import numpy as np

def safe_reduction(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    if np.any(np.abs(arr) > 65504.0):
        return float(np.sum(arr.astype(np.float64)))
    fp16_approx = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
    return float(np.sum(fp16_approx.astype(np.float32)))
