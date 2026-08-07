def calc_cache_bytes(seq_len, num_heads, head_dim, num_layers, policy):
    elements = seq_len * num_heads * head_dim * num_layers
    if policy == "fp16":
        return elements * 2
    if policy == "fp8_per_tensor":
        return elements + (num_layers * 4)
    if policy == "fp8_per_head":
        return elements + (num_layers * num_heads * 4)
    raise ValueError("Unknown policy")
