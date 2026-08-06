def compute_bytes_transferred(config: dict) -> dict:
    """Calculate naive and tiled HBM bytes transferred."""
    raise NotImplementedError


def compute_achieved_bandwidth(bytes_transferred: float, execution_time_sec: float) -> float:
    """Compute achieved memory bandwidth in GB/s."""
    raise NotImplementedError


def compute_bandwidth_efficiency(achieved_gbps: float, peak_gbps: float) -> float:
    """Compute bandwidth utilization ratio relative to theoretical peak."""
    raise NotImplementedError
