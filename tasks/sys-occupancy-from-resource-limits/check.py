import math

def _ref(regs_per_thread, smem_per_block, block_size,
         max_regs, max_smem, max_warps, max_blocks):
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

def grade(sol, fx) -> dict:
    cases = [
        # (regs_per_thread, smem_per_block, block_size,
        #  max_regs, max_smem, max_warps, max_blocks)
        (32, 0, 128, 65536, 98304, 64, 48),
        (64, 0, 256, 65536, 98304, 64, 48),
        (16, 32768, 128, 65536, 98304, 64, 48),
        (8, 1024, 32, 65536, 98304, 64, 48),
        (1, 0, 1024, 65536, 98304, 64, 48),
        (32, 0, 100, 65536, 98304, 64, 48),
        (32, 4096, 256, 32768, 49152, 32, 16),
        (16, 0, 32, 65536, 98304, 64, 48),
        (255, 0, 128, 65536, 98304, 64, 48),
        (128, 49152, 256, 65536, 98304, 64, 48),
    ]
    ok = 1.0
    for args in cases:
        try:
            got = int(sol.max_active_warps(*args))
        except Exception:
            ok = 0.0
            break
        if got != _ref(*args):
            ok = 0.0
            break
    return {"exact_match": ok}
