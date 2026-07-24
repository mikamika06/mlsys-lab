import numpy as np


def expand_kv_heads(kv: np.ndarray, num_query_heads: int) -> np.ndarray:
    # TODO: wrong head expansion. tile repeats the head dimension pattern
    # instead of repeating each KV head into a contiguous query-head group.
    repeat = num_query_heads // kv.shape[1]
    return np.tile(kv, (1, repeat, 1, 1))
