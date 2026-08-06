def compute_decode_bandwidth_floor(
    batch_size,
    current_seqlens,
    num_kv_heads,
    head_dim,
    num_layers,
    dtype_bytes,
    memory_clock_ghz,
    bus_width_bits,
    achieved_time_ms,
):
    raise NotImplementedError
