import numpy as np


def expand_kv_heads(kv: np.ndarray, n_query_heads: int) -> np.ndarray:
    kv = np.asarray(kv, dtype=np.float64)
    kv_heads = kv.shape[1]
    repeat = n_query_heads // kv_heads
    return np.repeat(kv, repeat, axis=1)
