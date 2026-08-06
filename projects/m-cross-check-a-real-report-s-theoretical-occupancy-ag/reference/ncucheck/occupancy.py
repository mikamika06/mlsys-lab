import math

def compute_theoretical_occupancy(threads_per_block, regs_per_thread, smem_per_block, arch="sm_80"):
    max_threads = 2048
    max_blocks = 32
    max_regs = 65536
    max_smem = 163840

    block_limit_threads = max_threads // threads_per_block
    block_limit_blocks = max_blocks

    reg_alloc_unit = 256
    warps_per_block = math.ceil(threads_per_block / 32.0)
    regs_per_warp = math.ceil((regs_per_thread * 32) / reg_alloc_unit) * reg_alloc_unit
    regs_per_block = regs_per_warp * warps_per_block
    block_limit_regs = max_regs // regs_per_block if regs_per_block > 0 else max_blocks

    smem_alloc_unit = 1024
    smem_per_block_aligned = math.ceil(smem_per_block / smem_alloc_unit) * smem_alloc_unit
    block_limit_smem = max_smem // smem_per_block_aligned if smem_per_block_aligned > 0 else max_blocks

    limiting_blocks = min(block_limit_threads, block_limit_blocks, block_limit_regs, block_limit_smem)
    active_threads = limiting_blocks * threads_per_block
    occupancy = (active_threads / max_threads) * 100.0
    return float(occupancy)
