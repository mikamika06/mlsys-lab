def bytes_transferred(m: int, n: int, k: int, itemsize: int = 2) -> int:
    """Calculate total byte transfers for matrix multiplication."""
    return (m * k + k * n + m * n) * itemsize


def achieved_bandwidth_gbps(total_bytes: int, elapsed_seconds: float) -> float:
    """Calculate achieved memory bandwidth in GB/s."""
    return (total_bytes / 1e9) / elapsed_seconds


def bandwidth_utilization_pct(achieved_gbps: float, peak_gbps: float = 546.0) -> float:
    """Calculate memory bandwidth utilization percentage against peak."""
    return (achieved_gbps / peak_gbps) * 100.0
