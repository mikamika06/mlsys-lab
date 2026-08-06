def bytes_transferred(m: int, n: int, k: int, itemsize: int = 2) -> int:
    """Calculate total byte transfers for matrix multiplication."""
    raise NotImplementedError


def achieved_bandwidth_gbps(total_bytes: int, elapsed_seconds: float) -> float:
    """Calculate achieved memory bandwidth in GB/s."""
    raise NotImplementedError


def bandwidth_utilization_pct(achieved_gbps: float, peak_gbps: float = 546.0) -> float:
    """Calculate memory bandwidth utilization percentage against peak."""
    raise NotImplementedError
