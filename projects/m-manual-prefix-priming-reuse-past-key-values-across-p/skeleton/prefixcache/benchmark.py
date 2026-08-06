def benchmark_cache_implementations(
    num_layers,
    num_kv_heads,
    head_dim,
    batch_size,
    max_seq_len,
    current_seq_len,
    dtype_bytes=2,
    quant_bits=8,
    offload_gpu_fraction=0.2,
):
    raise NotImplementedError
