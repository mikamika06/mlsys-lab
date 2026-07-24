import numpy as np

def kv_cache_bytes(
    layers: int,
    ctx_len: int,
    d_model: int,
    num_heads: int,
    groups: int = 1,
    dtype=np.float32,
) -> dict:
    """
    Compute the total KV cache size in bytes for MHA, GQA and MQA.
    Uses NumPy arrays to obtain the exact byte counts.
    """
    d_head = d_model // num_heads

    # Multi‑Head Attention
    k_mha = np.empty((layers, ctx_len, num_heads, d_head), dtype=dtype)
    v_mha = np.empty_like(k_mha)
    size_mha = (k_mha.nbytes + v_mha.nbytes)

    # Grouped‑Query Attention
    k_gqa = np.empty((layers, ctx_len, groups, d_head), dtype=dtype)
    v_gqa = np.empty_like(k_gqa)
    size_gqa = (k_gqa.nbytes + v_gqa.nbytes)

    # Multi‑Query Attention
    k_mqa = np.empty((layers, ctx_len, d_head), dtype=dtype)
    v_mqa = np.empty_like(k_mqa)
    size_mqa = (k_mqa.nbytes + v_mqa.nbytes)

    return {"mha": int(size_mha), "gqa": int(size_gqa), "mqa": int(size_mqa)}
