def calculate_wasted_lane_time(n: int, block_size: int, launch_overhead: float) -> float:
    """Calculates total wasted lane-time for a specific block size."""
    num_blocks = (n + block_size - 1) // block_size
    remainder = n % block_size
    padding_lanes = (block_size - remainder) if remainder != 0 else 0
    tail_waste = padding_lanes
    fixed_overhead = num_blocks * launch_overhead
    return float(tail_waste + fixed_overhead)
