def compute_occupancy(
    regs_per_thread: int,
    shared_bytes_per_block: int,
    threads_per_block: int,
    max_regs_per_sm: int,
    max_shared_bytes_per_sm: int,
    max_threads_per_sm: int,
    max_blocks_per_sm: int,
) -> tuple[int, float]:
    """Return (active_warps_per_sm, occupancy_fraction) -- see task.md."""
    raise NotImplementedError('your code here')
