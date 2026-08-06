def max_sequence_length(batch_size: int, num_heads: int, head_dim: int, memory_budget_bytes: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Find maximum integer sequence length within memory budget."""
    raise NotImplementedError


def max_batch_size(seq_len: int, num_heads: int, head_dim: int, memory_budget_bytes: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Find maximum integer batch size within memory budget."""
    raise NotImplementedError
