import numpy as np


def mha_to_gqa_pool(K: np.ndarray, V: np.ndarray, n_kv_heads: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert MHA key/value tensors to GQA-init key/value tensors by
    mean-pooling contiguous groups of original KV heads.

    K, V: (B, H, T, D) float64 MHA key/value tensors.
    n_kv_heads: target number of GQA KV heads G (H must be divisible by G).

    Heads are grouped contiguously in index order: heads
    [0, H/G) -> group 0, [H/G, 2H/G) -> group 1, etc.

    Returns (K_gqa, V_gqa), each of shape (B, G, T, D).
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    B, H, T, D = K.shape
    r = H // n_kv_heads

    K_gqa = K.reshape(B, n_kv_heads, r, T, D).mean(axis=2)
    V_gqa = V.reshape(B, n_kv_heads, r, T, D).mean(axis=2)
    return K_gqa, V_gqa
