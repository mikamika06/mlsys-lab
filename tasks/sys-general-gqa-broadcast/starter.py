import numpy as np


def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Grouped-query attention: query head h attends to KV head h // g, where
    g = n_q // n_kv (query heads split into n_kv contiguous groups of size
    g, each group sharing one KV head).

    Q: (n_q, n, d), K, V: (n_kv, n, d), n_q divisible by n_kv.
    Returns O: (n_q, n, d).
    """
    raise NotImplementedError('your code here')
