def compute_ep_all_to_all_volume(world_size, num_tokens, hidden_size, top_k, dtype_size):
    total_elements = num_tokens * top_k * hidden_size
    per_rank_send = (total_elements * dtype_size) * (world_size - 1) / world_size
    return float(per_rank_send)
