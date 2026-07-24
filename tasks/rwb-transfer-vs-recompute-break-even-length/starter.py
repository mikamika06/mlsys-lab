def break_even_length(
    kv_bytes_per_token: float,
    bandwidth: float,
    latency: float,
    prefill_throughput: float,
) -> float:
    """Return the sequence length where KV-transfer equals prefill recompute."""
    raise NotImplementedError("derive the break-even formula and implement it")
