def compute_activation_memory(batch_size: int, seq_len: int, num_heads: int, head_dim: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Compute backward activation memory in bytes."""
    base_tensors = 4 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    if mode == "prob":
        extra = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    elif mode == "lse":
        extra = batch_size * num_heads * seq_len * dtype_bytes
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return base_tensors + extra
