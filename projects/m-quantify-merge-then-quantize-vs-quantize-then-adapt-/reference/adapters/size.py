def compute_adapter_size(rank, module_set, hidden_dim, intermediate_dim, dtype_size):
    total_bytes = 0
    for module in module_set:
        if "gate" in module or "up" in module or "down" in module:
            dim1 = intermediate_dim
            dim2 = hidden_dim
        else:
            dim1 = hidden_dim
            dim2 = hidden_dim
        params = rank * (dim1 + dim2)
        total_bytes += params * dtype_size
    return total_bytes
