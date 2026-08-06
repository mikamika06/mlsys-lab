import numpy as np

def fused_rope_qk(
    Q: np.ndarray,
    K: np.ndarray,
    sin: np.ndarray,
    cos: np.ndarray
) -> np.ndarray:
    """
    Explicitly rotate queries and keys with RoPE and compute the dot product.
    This reference implementation is fully vectorised and serves as the oracle
    for grading.  It does *not* fuse the rotation into the score computation;
    it materialises rotated tensors first, which is correct but less efficient.
    """
    batch_size, seq_len_q, dim = Q.shape
    _, seq_len_k, _ = K.shape
    half = dim // 2

    out = np.empty((batch_size, seq_len_q, seq_len_k), dtype=Q.dtype)

    for b in range(batch_size):
        for l in range(seq_len_q):
            for m in range(seq_len_k):
                acc = 0.0
                for d in range(dim):
                    if d < half:
                        i = d
                        q_even = Q[b, l, 2 * i]
                        q_odd = Q[b, l, 2 * i + 1]
                        s_q = sin[l, i]
                        c_q = cos[l, i]
                        q_rot = q_even * c_q - q_odd * s_q

                        k_even = K[b, m, 2 * i]
                        k_odd = K[b, m, 2 * i + 1]
                        s_k = sin[m, i]
                        c_k = cos[m, i]
                        k_rot = k_even * c_k - k_odd * s_k

                        acc += q_rot * k_rot
                    else:
                        i = d - half
                        q_even = Q[b, l, 2 * i]
                        q_odd = Q[b, l, 2 * i + 1]
                        s_q = sin[l, i]
                        c_q = cos[l, i]
                        q_rot = q_even * s_q + q_odd * c_q

                        k_even = K[b, m, 2 * i]
                        k_odd = K[b, m, 2 * i + 1]
                        s_k = sin[m, i]
                        c_k = cos[m, i]
                        k_rot = k_even * s_k + k_odd * c_k

                        acc += q_rot * k_rot
                out[b, l, m] = acc

    return out
