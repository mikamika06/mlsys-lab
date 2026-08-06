"""Occupancy calculation reference."""

import math
import ref

def compute_theoretical_occupancy(kernel, device="sm_80"):
    spec = ref.DEVICE_SPECS[device]
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
