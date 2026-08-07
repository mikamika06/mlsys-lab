import numpy as np

def safe_all_reduce_sum(tensors: list[np.ndarray]) -> np.ndarray:
    res = np.zeros_like(tensors[0], dtype=np.float32)
    for t in tensors:
        res += t.astype(np.float32)
    return res
