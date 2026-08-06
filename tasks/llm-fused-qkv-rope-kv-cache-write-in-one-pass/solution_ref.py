import math
import numpy as np

def fused_qkv_rope_kv_cache_write(
    x: np.ndarray,
    weight_q: np.ndarray,
    weight_k: np.ndarray,
    weight_v: np.ndarray,
    rope_freqs: np.ndarray,
    kv_cache_k: np.ndarray,
    kv_cache_v: np.ndarray,
    cache_pos: int
):
    """
    Compute Q, K, V projections, apply RoPE to K and V, write rotated keys/values into the KV cache in place,
    and return (Q, K_rot, V_rot).
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    d_in = x.shape[2]

    d_q = weight_q.shape[1]
    d_k = weight_k.shape[1]
    d_v = weight_v.shape[1]

    q = np.empty((batch_size, seq_len, d_q), dtype=x.dtype)
    k_rot = np.empty((batch_size, seq_len, d_k), dtype=x.dtype)
    v_rot = np.empty((batch_size, seq_len, d_v), dtype=x.dtype)

    half_d_k = d_k // 2
    half_d_v = d_v // 2

    for b in range(batch_size):
        for s in range(seq_len):
            for j in range(d_q):
                acc = 0.0
                for m in range(d_in):
                    acc += x[b, s, m] * weight_q[m, j]
                q[b, s, j] = acc

            for j_half in range(half_d_k):
                angle = float(s) * float(rope_freqs[j_half])
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)

                idx_even = 2 * j_half
                idx_odd = 2 * j_half + 1

                k_even = 0.0
                k_odd = 0.0
                for m in range(d_in):
                    k_even += x[b, s, m] * weight_k[m, idx_even]
                    k_odd += x[b, s, m] * weight_k[m, idx_odd]

                rot_even = k_even * cos_val - k_odd * sin_val
                rot_odd = k_even * sin_val + k_odd * cos_val

                k_rot[b, s, idx_even] = rot_even
                k_rot[b, s, idx_odd] = rot_odd

                pos_idx = cache_pos + s
                kv_cache_k[b, pos_idx, idx_even] = rot_even
                kv_cache_k[b, pos_idx, idx_odd] = rot_odd

            for j_half in range(half_d_v):
                angle = float(s) * float(rope_freqs[j_half])
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)

                idx_even = 2 * j_half
                idx_odd = 2 * j_half + 1

                v_even = 0.0
                v_odd = 0.0
                for m in range(d_in):
                    v_even += x[b, s, m] * weight_v[m, idx_even]
                    v_odd += x[b, s, m] * weight_v[m, idx_odd]

                rot_even = v_even * cos_val - v_odd * sin_val
                rot_odd = v_even * sin_val + v_odd * cos_val

                v_rot[b, s, idx_even] = rot_even
                v_rot[b, s, idx_odd] = rot_odd

                pos_idx = cache_pos + s
                kv_cache_v[b, pos_idx, idx_even] = rot_even
                kv_cache_v[b, pos_idx, idx_odd] = rot_odd

    return q, k_rot, v_rot
