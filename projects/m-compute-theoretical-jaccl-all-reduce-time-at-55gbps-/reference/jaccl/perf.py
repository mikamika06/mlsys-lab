import math


def compute_allreduce_time(tensor_bytes: int, num_ranks: int, bandwidth_gbps: float = 55.0) -> float:
    """Calculates theoretical ring all-reduce transfer time in seconds."""
    bytes_transferred = 2.0 * (num_ranks - 1) / num_ranks * tensor_bytes
    bw_bytes_sec = (bandwidth_gbps * 1e9) / 8.0
    return bytes_transferred / bw_bytes_sec


def compute_allreduce_overhead(
    tensor_bytes: int,
    num_ranks: int,
    latency_per_step_sec: float,
    compute_time_sec: float,
    bandwidth_gbps: float = 55.0,
) -> float:
    """Calculates ring all-reduce latency overhead relative to single-process compute time."""
    transfer_time = compute_allreduce_time(tensor_bytes, num_ranks, bandwidth_gbps)
    hop_latency = 2.0 * (num_ranks - 1) * latency_per_step_sec
    total_comm = transfer_time + hop_latency
    return total_comm / compute_time_sec


def min_microbatches_for_bubble(num_ranks: int, target_bubble_fraction: float) -> int:
    """Calculates minimum microbatch count to achieve <= target bubble fraction."""
    if target_bubble_fraction <= 0 or target_bubble_fraction >= 1:
        raise ValueError("Target bubble fraction must be between 0 and 1.")
    m = ((num_ranks - 1) * (1.0 - target_bubble_fraction)) / target_bubble_fraction
    return max(1, math.ceil(m - 1e-9))
