def compute_latency_stats(unfused_ms, fused_ms):
    """Compute latency differences and speedup ratios."""
    raise NotImplementedError


def compute_throughput_gbs(ms_latency, total_bytes):
    """Compute effective memory throughput in GB/s."""
    raise NotImplementedError
