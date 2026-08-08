def classify_kernel(ai: float, ridge_point: float) -> str:
    """Classify kernel as compute-bound or memory-bound based on AI and ridge point."""
    raise NotImplementedError


def max_achievable_gflops(ai: float, peak_gflops: float, bandwidth_gbps: float) -> float:
    """Compute the roofline upper bound for performance in GFLOP/s."""
    raise NotImplementedError
