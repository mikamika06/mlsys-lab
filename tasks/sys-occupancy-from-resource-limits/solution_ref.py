import math

def max_active_warps(
    regs_per_thread: int,
    smem_per_block: int,
    block_size: int,
    max_regs: int = 65536,
    max_smem: int = 98304,
    max_warps: int = 64,
    max_blocks: int = 48,
) -> int:
    WARP_SIZE = 32
    warps_per_block = math.ceil(block_size / WARP_SIZE)
    regs_per_block = block_size * regs_per_thread

    if regs_per_block == 0:
        blocks_by_regs = max_blocks
    else:
        blocks_by_regs = max_regs // regs_per_block

    if smem_per_block == 0:
        blocks_by_smem = max_blocks
    else:
        blocks_by_smem = max_smem // smem_per_block

    effective_blocks = min(blocks_by_regs, blocks_by_smem, max_blocks)
    return min(effective_blocks * warps_per_block, max_warps)
