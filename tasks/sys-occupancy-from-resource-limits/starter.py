def max_active_warps(
    regs_per_thread: int,
    smem_per_block: int,
    block_size: int,
    max_regs: int = 65536,
    max_smem: int = 98304,
    max_warps: int = 64,
    max_blocks: int = 48,
) -> int:
    """Return the maximum number of active warps per SM given resource limits."""
    raise NotImplementedError("your code here")
