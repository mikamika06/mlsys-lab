from typing import List


class LatencyTracker:
    """Tracks per-step latency and evaluates ITL spike metrics."""

    def __init__(self) -> None:
        raise NotImplementedError

    def record_step(self, latency_ms: float) -> None:
        raise NotImplementedError

    def get_latencies(self) -> List[float]:
        raise NotImplementedError

    def compute_max_to_avg_ratio(self) -> float:
        raise NotImplementedError
