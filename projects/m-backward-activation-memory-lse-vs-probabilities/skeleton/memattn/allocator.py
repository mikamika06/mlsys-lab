def compute_activation_memory(batch_size: int, seq_len: int, num_heads: int, head_dim: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Compute backward activation memory in bytes."""
    raise NotImplementedError
