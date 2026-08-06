def compute_theoretical_occupancy(regs_per_thread: int, smem_per_block: int, block_size: int, device_props: dict) -> float:
    max_threads_per_sm = device_props.get("max_threads_per_sm", 2048)
    max_blocks_per_sm = device_props.get("max_blocks_per_sm", 32)
    max_registers_per_sm = device_props.get("max_registers_per_sm", 65536)
    max_smem_per_sm = device_props.get("max_smem_per_sm", 49152)
    reg_alloc_granularity = device_props.get("reg_alloc_granularity", 256)
    warp_size = device_props.get("warp_size", 32)

    warps_per_block = (block_size + warp_size - 1) // warp_size
    threads_per_block = warps_per_block * warp_size

    if threads_per_block > max_threads_per_sm:
        return 0.0

    limit_by_threads = max_threads_per_sm // threads_per_block

    if reg_alloc_granularity > 0:
        allocated_regs = ((regs_per_thread * warp_size + reg_alloc_granularity - 1) // reg_alloc_granularity) * reg_alloc_granularity
        total_regs_per_warp = allocated_regs
    else:
        total_regs_per_warp = regs_per_thread * warp_size

    regs_per_block_total = total_regs_per_warp * warps_per_block
    limit_by_regs = max_registers_per_sm // regs_per_block_total if regs_per_block_total > 0 else max_blocks_per_sm

    if smem_per_block > 0:
        limit_by_smem = max_smem_per_sm // smem_per_block
    else:
        limit_by_smem = max_blocks_per_sm

    limit_by_blocks = max_blocks_per_sm

    active_blocks = min(limit_by_threads, limit_by_regs, limit_by_smem, limit_by_blocks)
    active_warps = active_blocks * warps_per_block
    max_possible_warps = max_threads_per_sm // warp_size

    occupancy = float(active_warps) / float(max_possible_warps)
    return float(min(1.0, max(0.0, occupancy)))
