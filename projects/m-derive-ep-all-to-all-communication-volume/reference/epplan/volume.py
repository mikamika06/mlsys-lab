def compute_all_to_all_volume(num_ranks: int, seq_len: int, hidden_dim: int, top_k: int, dtype_size: int) -> int:
    tokens_per_rank = seq_len
    total_tokens = tokens_per_rank * num_ranks
    dispatched_tokens = total_tokens * top_k
    bytes_per_token = hidden_dim * dtype_size
    total_bytes = dispatched_tokens * bytes_per_token
    return total_bytes
