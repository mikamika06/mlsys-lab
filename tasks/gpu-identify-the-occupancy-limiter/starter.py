def identify_limiter(block_dim: int,
                     regs_per_thread: int,
                     shared_bytes_per_block: int) -> str:
    """Determine which resource limits GPU kernel occupancy."""
    raise NotImplementedError('your code here')
