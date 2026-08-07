from occupancy.calc import compute_theoretical_occupancy

def optimize_register_cap(threads_per_block, smem_per_block, spill_threshold_regs, limits):
    best_regs = 16
    best_occ = -1.0
    for r in range(16, spill_threshold_regs + 1, 4):
        res = compute_theoretical_occupancy(threads_per_block, r, smem_per_block, limits)
        if res["occupancy"] > best_occ:
            best_occ = res["occupancy"]
            best_regs = r
    return best_regs
