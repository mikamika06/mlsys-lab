def should_quantize_kv(seq_len: int, batch_size: int, kv_heads: int, head_dim: int, mse_threshold: float) -> bool:
    """Decide whether to quantize KV cache based on workload size and error tolerance."""
    footprint = seq_len * batch_size * kv_heads * head_dim * 2
    if footprint < 1024 * 1024:
        return False
    if mse_threshold < 0.0001:
        return False
    return True
