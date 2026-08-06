from triton_calc.regs import effective_regs


def max_resident_blocks(
    regs_per_thread: int, threads_per_block: int, spec: dict
) -> int:
    """Compute maximum resident blocks per SM given register pressure and SM specs."""
    gran = spec.get("reg_granularity", 8)
    eff = effective_regs(regs_per_thread, gran)
    block_regs = eff * threads_per_block
    if block_regs == 0:
        by_regs = spec["max_blocks_per_sm"]
    else:
        by_regs = spec["max_regs_per_sm"] // block_regs
    by_threads = spec["max_threads_per_sm"] // threads_per_block
    return int(min(by_regs, by_threads, spec["max_blocks_per_sm"]))
