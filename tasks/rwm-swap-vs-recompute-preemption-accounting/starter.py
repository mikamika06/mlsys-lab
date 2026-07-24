import math

def preemption_costs(
    seq_len: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    """Return (swap_cost_bytes, recompute_cost_tokens)."""
    num_blocks = seq_len // block_size          # BUG: truncates instead of ceil
    per_token = num_layers * num_kv_heads * head_dim * dtype_bytes
    swap_cost = num_blocks * block_size * per_token  # BUG: missing round-trip factor of 2
    recompute_cost = seq_len
    return (swap_cost, recompute_cost)
