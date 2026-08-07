def compute_comm_volume(num_tokens, world_size, head_dim, dtype_size):
    tokens_per_rank = num_tokens // world_size
    bytes_per_token = head_dim * dtype_size
    steps = world_size - 1
    total_bytes = tokens_per_rank * bytes_per_token * steps
    return total_bytes
