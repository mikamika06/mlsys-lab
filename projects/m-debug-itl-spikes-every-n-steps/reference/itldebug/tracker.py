from typing import List


class LatencyTracker:
    """Tracks per-step latency and evaluates ITL spike metrics."""

    def __init__(self) -> None:
        self.latencies: List[float] = []

    def record_step(self, latency_ms: float) -> None:
        self.latencies.append(float(latency_ms))

    def get_latencies(self) -> List[float]:
        return list(self.latencies)

    def compute_max_to_avg_ratio(self) -> float:
        if not self.latencies:
            return 1.0
        avg_lat = sum(self.latencies) / len(self.latencies)
        if avg_lat <= 1e-9:
            return 1.0
        return max(self.latencies) / avg_lat
