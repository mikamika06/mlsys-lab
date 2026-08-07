def communication_volume(num_tokens, hidden_dim, num_experts, top_k, world_size, capacity, bytes_per_elem=4):
    """
    Calculate MoE all-to-all vs Dense FFN all-reduce communication volume in bytes.
    """
    raise NotImplementedError
