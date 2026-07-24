import numpy as np


def expand_kv_heads(kv: np.ndarray, num_q_heads: int) -> np.ndarray:
    groups = num_q_heads // kv.shape[1]
    return np.repeat(kv, groups, axis=1)
