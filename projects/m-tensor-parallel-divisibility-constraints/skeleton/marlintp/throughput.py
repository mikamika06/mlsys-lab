def estimate_throughput(
    cfg: dict,
    batch_size: int,
    seq_len: int,
    memory_bandwidth_gbps: float = 900.0,
    compute_tflops: float = 312.0,
) -> dict:
    raise NotImplementedError
