import math

ARCH_SPEC = {
    "max_regs_per_sm": 65536,
    "max_smem_per_sm": 102400,
    "max_warps_per_sm": 64,
    "max_blocks_per_sm": 32,
    "reg_alloc_granularity": 256,
    "warp_size": 32,
}

CONFIGS = [
    {"block_size": 256, "regs_per_thread": 32, "smem_bytes": 1024, "spill_limit": 48},
    {"block_size": 512, "regs_per_thread": 64, "smem_bytes": 16384, "spill_limit": 64},
    {"block_size": 128, "regs_per_thread": 16, "smem_bytes": 4096, "spill_limit": 32},
    {"block_size": 1024, "regs_per_thread": 24, "smem_bytes": 32768, "spill_limit": 32},
    {"block_size": 256, "regs_per_thread": 48, "smem_bytes": 8192, "spill_limit": 56},
]

NCU_REPORTS = [
    {"id": 1, "theoretical_occupancy": 0.75, "achieved_occupancy": 0.70, "spills": 0},
    {"id": 2, "theoretical_occupancy": 0.50, "achieved_occupancy": 0.48, "spills": 0},
    {"id": 3, "theoretical_occupancy": 1.00, "achieved_occupancy": 0.85, "spills": 0},
    {"id": 4, "theoretical_occupancy": 0.375, "achieved_occupancy": 0.35, "spills": 128},
    {"id": 5, "theoretical_occupancy": 0.625, "achieved_occupancy": 0.60, "spills": 0},
]


def compute_theoretical_occupancy(cfg):
    bs = cfg["block_size"]
    rpt = cfg["regs_per_thread"]
    smem = cfg["smem_bytes"]
    warp_size = ARCH_SPEC["warp_size"]
    warps_per_block = math.ceil(bs / warp_size)
    regs_per_warp = math.ceil((rpt * warp_size) / ARCH_SPEC["reg_alloc_granularity"]) * ARCH_SPEC["reg_alloc_granularity"]
    regs_per_block = regs_per_warp * warps_per_block
    max_blocks_regs = ARCH_SPEC["max_regs_per_sm"] // regs_per_block if regs_per_block > 0 else 0
    max_blocks_smem = ARCH_SPEC["max_smem_per_sm"] // smem if smem > 0 else ARCH_SPEC["max_blocks_per_sm"]
    max_blocks_warps = ARCH_SPEC["max_warps_per_sm"] // warps_per_block
    max_blocks_limit = ARCH_SPEC["max_blocks_per_sm"]
    blocks_per_sm = min(max_blocks_regs, max_blocks_smem, max_blocks_warps, max_blocks_limit)
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / ARCH_SPEC["max_warps_per_sm"]
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


def cross_check_ncu(report, computed_occ):
    return abs(report["theoretical_occupancy"] - computed_occ) < 0.05
