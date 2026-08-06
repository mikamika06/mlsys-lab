def estimate_tp_layer_comm_bytes(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    ffn_dim: int,
    tp_size: int,
    element_size_bytes: int = 2,
) -> dict:
    if tp_size <= 1:
        return {
            "forward_all_reduce_bytes": 0,
            "backward_all_reduce_bytes": 0,
            "total_bytes_per_step": 0,
        }

    tokens = batch_size * seq_len
    fwd_bytes_per_ar = 2 * ((tp_size - 1) / tp_size) * tokens * hidden_dim * element_size_bytes
    bwd_bytes_per_ar = 2 * ((tp_size - 1) / tp_size) * tokens * hidden_dim * element_size_bytes

    total = fwd_bytes_per_ar + bwd_bytes_per_ar
    return {
        "forward_all_reduce_bytes": int(fwd_bytes_per_ar),
        "backward_all_reduce_bytes": int(bwd_bytes_per_ar),
        "total_bytes_per_step": int(total),
    }
