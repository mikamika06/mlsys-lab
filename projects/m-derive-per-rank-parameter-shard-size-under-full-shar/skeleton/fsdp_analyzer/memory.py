def compute_layer_transient_peak_memory_bytes(
    layer_params: int,
    world_size: int,
    rank: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    bytes_per_param: int = 2,
    bytes_per_activation: int = 2,
) -> int:
    """Compute transient peak memory during one layer forward pass for a rank under FULL_SHARD."""
    raise NotImplementedError
