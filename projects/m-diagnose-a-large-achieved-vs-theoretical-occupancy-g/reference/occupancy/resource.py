import numpy as np

def compute_theoretical_occupancy(config, limits):
    regs_per_thread = config["regs_per_thread"]
    threads_per_block = config["threads_per_block"]
    smem_per_block = config["smem_per_block"]

    max_threads_sm = limits["max_threads_per_sm"]
    max_regs_sm = limits["max_regs_per_sm"]
    max_smem_sm = limits["max_smem_sm"]
    allocation_granularity_regs = limits.get("reg_allocation_unit", 256)
    allocation_granularity_threads = limits.get("thread_allocation_unit", 32)

    alloc_threads = ((threads_per_block + allocation_granularity_threads - 1) // allocation_granularity_threads) * allocation_granularity_threads
    blocks_by_threads = max_threads_sm // alloc_threads

    regs_per_warp = ((regs_per_thread * 32 + allocation_granularity_regs - 1) // allocation_granularity_regs) * allocation_granularity_regs
    total_regs_block = regs_per_warp * (threads_per_block // 32)
    blocks_by_regs = max_regs_sm // total_regs_block if total_regs_block > 0 else 999

    blocks_by_smem = max_smem_sm // smem_per_block if smem_per_block > 0 else 999

    max_blocks_per_sm = limits.get("max_blocks_per_sm", 32)
    active_blocks = min(blocks_by_threads, blocks_by_regs, blocks_by_smem, max_blocks_per_sm)
    active_threads = active_blocks * threads_per_block
    return float(active_threads) / float(max_threads_sm)

def rank_kernels(configs, limits):
    scored = []
    for cfg in configs:
        occ = compute_theoretical_occupancy(cfg, limits)
        scored.append((cfg["id"], occ))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored]
