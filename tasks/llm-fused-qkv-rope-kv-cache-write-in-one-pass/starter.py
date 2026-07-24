import numpy as np

def fused_qkv_rope_kv_cache_write(x: np.ndarray, weight_q: np.ndarray, weight_k: np.ndarray, weight_v: np.ndarray, rope_freqs: np.ndarray, kv_cache_k: np.ndarray, kv_cache_v: np.ndarray, cache_pos: int):
    """Broken implementation: applies RoPE only to keys, writes values unchanged."""
    raise NotImplementedError('your code here')
