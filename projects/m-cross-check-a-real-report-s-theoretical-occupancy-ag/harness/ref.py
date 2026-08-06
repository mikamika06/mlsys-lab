import math

DEVICE_SPECS = {
    "sm_80": {
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_registers_per_sm": 65536,
        "max_shared_mem_per_sm": 163840,
        "reg_allocation_granularity": 256,
        "warp_size": 32,
        "max_threads_per_block": 1024,
    }
}

FIELDS_TO_SECTIONS = {
    "smsp__cycles_active.sum": "SpeedOfLight",
    "sm__warps_active.avg.per_cycle_active": "Occupancy",
    "launch__threads_per_block": "LaunchStats",
    "launch__registers_per_thread": "LaunchStats",
    "launch__shared_mem_per_block": "LaunchStats",
}

KERNELS = [
    {
        "threads_per_block": 256,
        "registers_per_thread": 32,
        "shared_mem_per_block": 1024,
    },
    {
        "threads_per_block": 512,
        "registers_per_thread": 64,
        "shared_mem_per_block": 4096,
    },
    {
        "threads_per_block": 128,
        "registers_per_thread": 16,
        "shared_mem_per_block": 0,
    }
]

def compute_theoretical_occupancy(kernel, device="sm_80"):
    spec = DEVICE_SPECS[device]
    tpbl = kernel["threads_per_block"]
    rpt = kernel["registers_per_thread"]
    smem = kernel["shared_mem_per_block"]
    warp_size = spec["warp_size"]
    warps_per_block = math.ceil(tpbl / warp_size)
    blocks_by_regs = spec["max_registers_per_sm"] // max(1, tpbl * rpt)
    blocks_by_threads = spec["max_threads_per_sm"] // tpbl
    blocks_by_smem = spec["max_shared_mem_per_sm"] // max(1, smem) if smem > 0 else spec["max_blocks_per_sm"]
    active_blocks = min(spec["max_blocks_per_sm"], blocks_by_threads, blocks_by_regs, blocks_by_smem)
    active_warps = active_blocks * warps_per_block
    max_warps = spec["max_threads_per_sm"] // warp_size
    return float((active_warps / max_warps) * 100.0)
