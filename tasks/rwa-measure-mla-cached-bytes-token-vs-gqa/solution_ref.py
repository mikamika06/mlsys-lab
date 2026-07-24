import numpy as np

def cached_bytes_per_token(
    n_kv_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    dtype: np.dtype
) -> tuple[int, int]:
    """
    Return (mla_bytes_per_token, gqa_bytes_per_token).
    """
    bytes_per_element = np.dtype(dtype).itemsize
    mla_bytes = kv_lora_rank * bytes_per_element
    gqa_bytes = 2 * n_kv_heads * head_dim * bytes_per_element
    return (mla_bytes, gqa_bytes)
