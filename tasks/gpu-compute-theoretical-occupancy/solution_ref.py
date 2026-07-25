WARP = 32


def compute_occupancy(
    regs_per_thread: int,
    shared_bytes_per_block: int,
    threads_per_block: int,
    max_regs_per_sm: int,
    max_shared_bytes_per_sm: int,
    max_threads_per_sm: int,
    max_blocks_per_sm: int,
) -> tuple[int, float]:
    """Return (active_warps_per_sm, occupancy_fraction).

    A block can be resident on the SM only if there is room for it under
    EVERY resource limit at once, so the number of concurrently resident
    blocks is the MINIMUM of what each individual limit alone would
    allow, plus the hardware's own flat block-count cap.
    """
    blocks_by_regs = max_regs_per_sm // (regs_per_thread * threads_per_block)
    if shared_bytes_per_block > 0:
        blocks_by_shared = max_shared_bytes_per_sm // shared_bytes_per_block
    else:
        blocks_by_shared = max_blocks_per_sm
    blocks_by_threads = max_threads_per_sm // threads_per_block

    active_blocks = min(blocks_by_regs, blocks_by_shared, blocks_by_threads, max_blocks_per_sm)

    warps_per_block = (threads_per_block + WARP - 1) // WARP
    active_warps = active_blocks * warps_per_block

    max_warps_per_sm = max_threads_per_sm // WARP
    occupancy = active_warps / max_warps_per_sm

    return active_warps, occupancy
