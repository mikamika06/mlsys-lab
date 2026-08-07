def should_quantize_kv(seq_len: int, batch_size: int, kv_heads: int, head_dim: int, mse_threshold: float) -> bool:
    """Decide whether to quantize KV cache."""
    raise NotImplementedError
