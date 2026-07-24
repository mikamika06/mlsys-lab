def identify_limiter(block_dim: int,
                     regs_per_thread: int,
                     shared_bytes_per_block: int) -> str:
    """
    Determine which resource limits the occupancy of a GPU kernel.

    Parameters
    ----------
    block_dim : int
        Number of threads per block.
    regs_per_thread : int
        Registers consumed by each thread.
    shared_bytes_per_block : int
        Bytes of shared memory requested by each block.

    Returns
    -------
    str
        One of 'register', 'shared', or 'thread' indicating the limiting
        resource according to the typical GPU limits used in this task.
    """
    # Typical hardware limits for a modern NVIDIA GPU
    max_regs_per_sm = 65536          # registers per SM
    max_shared_bytes = 49152         # bytes of shared memory per SM (48 KiB)
    max_threads_per_sm = 2048        # maximum active threads per SM

    # Compute threads allowed by register usage
    threads_by_regs = min(max_regs_per_sm // regs_per_thread,
                          max_threads_per_sm)

    # Compute threads allowed by shared‑memory usage
    if shared_bytes_per_block <= 0:
        # If no shared memory is requested, shared memory never limits occupancy.
        blocks_by_shared = max_threads_per_sm
    else:
        blocks_by_shared = max(1, max_shared_bytes // shared_bytes_per_block)
    threads_by_shared = min(blocks_by_shared * block_dim,
                            max_threads_per_sm)

    # The hard thread‑cap is also a limiting factor
    threads_by_thread_cap = max_threads_per_sm

    limits = {
        'register': threads_by_regs,
        'shared':   threads_by_shared,
        'thread':   threads_by_thread_cap
    }

    # Identify the resource that yields the smallest allowed thread count.
    limiting_key = min(limits, key=limits.get)
    return limiting_key
