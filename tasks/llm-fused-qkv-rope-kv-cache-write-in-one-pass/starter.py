import math

def fused_qkv_rope_kv_cache_write(
    x: list[list[list[float]]],
    weight_q: list[list[float]],
    weight_k: list[list[float]],
    weight_v: list[list[float]],
    rope_freqs: list[float],
    kv_cache_k: list[list[list[float]]],
    kv_cache_v: list[list[list[float]]],
    cache_pos: int
) -> Tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]:
    """Broken implementation: applies RoPE only to keys, writes values unchanged."""
    raise NotImplementedError('your code here')
