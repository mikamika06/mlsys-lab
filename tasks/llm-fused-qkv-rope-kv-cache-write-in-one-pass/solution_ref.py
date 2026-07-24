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

    # Linear projections
    q = x @ weight_q
    k = x @ weight_k
    v = x @ weight_v

    seq_len = x.shape[1]
    d_model = x.shape[2]

    # RoPE angles
    pos = np.arange(seq_len, dtype=np.float64)[:, None]          # (seq_len,1)
    freq = rope_freqs[None, :]                                   # (1,d//2)
    angle = pos * freq                                           # (seq_len,d//2)
    cos = np.cos(angle)                                          # (seq_len,d//2)
    sin = np.sin(angle)                                          # (seq_len,d//2)

    # Expand to match batch and seq dimensions
    cos_tiled = cos[None, :, :]                                   # (1,seq_len,d//2)
    sin_tiled = sin[None, :, :]

    # Apply RoPE to K
    k_even = k[..., ::2]
    k_odd  = k[..., 1::2]
    k_rot = np.empty_like(k)
    k_rot[..., ::2] = k_even * cos_tiled - k_odd * sin_tiled
    k_rot[..., 1::2] = k_even * sin_tiled + k_odd * cos_tiled

    # Apply RoPE to V
    v_even = v[..., ::2]
    v_odd  = v[..., 1::2]
    v_rot = np.empty_like(v)
    v_rot[..., ::2] = v_even * cos_tiled - v_odd * sin_tiled
    v_rot[..., 1::2] = v_even * sin_tiled + v_odd * cos_tiled

    # Write into cache in place
    kv_cache_k[:, cache_pos:cache_pos+seq_len, :] = k_rot
    kv_cache_v[:, cache_pos:cache_pos+seq_len, :] = v_rot

    return q, k_rot, v_rot
