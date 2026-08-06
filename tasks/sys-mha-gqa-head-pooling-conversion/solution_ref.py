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

    K_gqa = np.empty((B, n_kv_heads, T, D), dtype=np.float64)
    V_gqa = np.empty((B, n_kv_heads, T, D), dtype=np.float64)

    for b in range(B):
        for g in range(n_kv_heads):
            for t in range(T):
                for d in range(D):
                    k_sum = 0.0
                    v_sum = 0.0
                    for i in range(r):
                        h = g * r + i
                        k_sum += K[b, h, t, d]
                        v_sum += V[b, h, t, d]
                    K_gqa[b, g, t, d] = k_sum / r
                    V_gqa[b, g, t, d] = v_sum / r

    return K_gqa, V_gqa
