import numpy as np


def expand_gqa_kv(kv: np.ndarray, num_query_heads: int) -> np.ndarray:
    repeat = num_query_heads // kv.shape[1]
    return np.repeat(kv, repeat, axis=1).astype(np.float32)
