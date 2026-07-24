def break_even_length(
    kv_bytes_per_token: float,
    bandwidth: float,
    latency: float,
    prefill_throughput: float,
) -> float:
    """Return the sequence length where KV-transfer equals prefill recompute.

    L* = latency / (1/prefill_throughput - kv_bytes_per_token/bandwidth)

    Returns float('inf') when recompute is always at least as fast.
    Returns 0.0 when transfer is strictly cheaper per token with zero latency.
    """
    denom = 1.0 / prefill_throughput - kv_bytes_per_token / bandwidth
    if denom <= 0:
        return float("inf")
    if latency == 0:
        return 0.0
    return latency / denom
