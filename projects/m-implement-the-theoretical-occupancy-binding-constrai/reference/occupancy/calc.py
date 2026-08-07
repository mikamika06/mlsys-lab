import math


def compute_theoretical_occupancy(cfg, arch_spec=None):
    if arch_spec is None:
        arch_spec = {
            "max_regs_per_sm": 65536,
            "max_smem_per_sm": 102400,
            "max_warps_per_sm": 64,
            "max_blocks_per_sm": 32,
            "reg_alloc_granularity": 256,
            "warp_size": 32,
        }
    bs = cfg["block_size"]
    rpt = cfg["regs_per_thread"]
    smem = cfg["smem_bytes"]
    warp_size = arch_spec["warp_size"]
    warps_per_block = math.ceil(bs / warp_size)
    regs_per_warp = math.ceil((rpt * warp_size) / arch_spec["reg_alloc_granularity"]) * arch_spec["reg_alloc_granularity"]
    regs_per_block = regs_per_warp * warps_per_block
    max_blocks_regs = arch_spec["max_regs_per_sm"] // regs_per_block if regs_per_block > 0 else 0
    max_blocks_smem = arch_spec["max_smem_per_sm"] // smem if smem > 0 else arch_spec["max_blocks_per_sm"]
    max_blocks_warps = arch_spec["max_warps_per_sm"] // warps_per_block
    max_blocks_limit = arch_spec["max_blocks_per_sm"]
    blocks_per_sm = min(max_blocks_regs, max_blocks_smem, max_blocks_warps, max_blocks_limit)
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / arch_spec["max_warps_per_sm"]
    constraints = {
        "registers": max_blocks_regs,
        "shared_memory": max_blocks_smem,
        "warps": max_blocks_warps,
        "blocks_limit": max_blocks_limit
    }
    binding = min(constraints, key=constraints.get)
    return float(occupancy), binding


def find_optimal_register_cap(cfg, spill_threshold):
    best_cap = cfg["regs_per_thread"]
    best_occ = 0.0
    for cap in range(16, spill_threshold + 1):
        test_cfg = dict(cfg, regs_per_thread=cap)
        occ, _ = compute_theoretical_occupancy(test_cfg)
        if occ > best_occ:
            best_occ = occ
            best_cap = cap
    return best_cap
