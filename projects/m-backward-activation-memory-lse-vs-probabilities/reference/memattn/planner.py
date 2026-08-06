import math
from memattn.allocator import compute_activation_memory


def max_sequence_length(batch_size: int, num_heads: int, head_dim: int, memory_budget_bytes: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Find maximum integer sequence length within memory budget."""
    if mode == "lse":
        per_token_bytes = batch_size * num_heads * (4 * head_dim + 1) * dtype_bytes
        return memory_budget_bytes // per_token_bytes
    elif mode == "prob":
        a = batch_size * num_heads * dtype_bytes
        b = 4 * batch_size * num_heads * head_dim * dtype_bytes
        c = -memory_budget_bytes
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0
        n = (-b + math.sqrt(disc)) / (2 * a)
        return int(math.floor(n))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def max_batch_size(seq_len: int, num_heads: int, head_dim: int, memory_budget_bytes: int, mode: str = "lse", dtype_bytes: int = 2) -> int:
    """Find maximum integer batch size within memory budget."""
    single_batch_mem = compute_activation_memory(1, seq_len, num_heads, head_dim, mode=mode, dtype_bytes=dtype_bytes)
    return memory_budget_bytes // single_batch_mem
