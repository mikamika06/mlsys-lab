def calculate_percentile(data: list[float], p: float, method: str = "nearest") -> float:
    """Calculate p-th percentile using nearest-rank or linear interpolation."""
    raise NotImplementedError


def decompose_latencies(requests: list[dict], method: str = "nearest") -> dict:
    """Decompose p95 E2E latency into queue, prefill, and decode components."""
    raise NotImplementedError
