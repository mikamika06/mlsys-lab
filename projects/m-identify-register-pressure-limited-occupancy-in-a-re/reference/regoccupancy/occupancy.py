def compute_occupancy(regs_per_thread, block_size, smem_bytes, arch):
    max_threads_per_sm = arch.get("max_threads_per_sm", 2048)
    max_blocks_per_sm = arch.get("max_blocks_per_sm", 32)
    max_regs_per_sm = arch.get("max_regs_per_sm", 65536)
    max_smem_per_sm = arch.get("max_smem_per_sm", 163840)
    reg_allocation_granularity = arch.get("reg_allocation_granularity", 256)
    warp_size = 32

    warps_per_block = (block_size + warp_size - 1) // warp_size
    threads_per_block = warps_per_block * warp_size

    if threads_per_block > max_threads_per_sm:
        return 0.0

    if reg_allocation_granularity > 1:
        regs_per_thread_rounded = ((regs_per_thread + reg_allocation_granularity - 1) // reg_allocation_granularity) * reg_allocation_granularity
    else:
        regs_per_thread_rounded = regs_per_thread

    regs_per_block = regs_per_thread_rounded * threads_per_block
    blocks_by_regs = max_regs_per_sm // regs_per_block if regs_per_block > 0 else max_blocks_per_sm

    if smem_bytes > 0:
        smem_granularity = arch.get("smem_granularity", 128)
        smem_rounded = ((smem_bytes + smem_granularity - 1) // smem_granularity) * smem_granularity
        blocks_by_smem = max_smem_per_sm // smem_rounded
    else:
        blocks_by_smem = max_blocks_per_sm

    blocks_by_threads = max_threads_per_sm // threads_per_block

    active_blocks = min(max_blocks_per_sm, blocks_by_regs, blocks_by_smem, blocks_by_threads)
    active_threads = active_blocks * threads_per_block
    occupancy = float(active_threads) / float(max_threads_per_sm)
    return occupancy
