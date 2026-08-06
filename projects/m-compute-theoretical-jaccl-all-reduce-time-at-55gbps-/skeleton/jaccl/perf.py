def compute_allreduce_time(tensor_bytes: int, num_ranks: int, bandwidth_gbps: float = 55.0) -> float:
    """Calculates theoretical ring all-reduce transfer time in seconds."""
    raise NotImplementedError


def compute_allreduce_overhead(
    tensor_bytes: int,
    num_ranks: int,
    latency_per_step_sec: float,
    compute_time_sec: float,
    bandwidth_gbps: float = 55.0,
) -> float:
    """Calculates ring all-reduce latency overhead relative to single-process compute time."""
    raise NotImplementedError


def min_microbatches_for_bubble(num_ranks: int, target_bubble_fraction: float) -> int:
    """Calculates minimum microbatch count to achieve <= target bubble fraction."""
    raise NotImplementedError
