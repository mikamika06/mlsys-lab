def measure_peak_inflight_microbatches(p: int, m: int, v: int = 1, schedule_type: str = "1f1b") -> list[int]:
    """Measure peak in-flight microbatches per physical rank."""
    raise NotImplementedError


def estimate_activation_memory_mb(peak_inflight: list[int], bytes_per_microbatch: float) -> list[float]:
    """Calculate activation memory per rank based on peak in-flight microbatches."""
    raise NotImplementedError
