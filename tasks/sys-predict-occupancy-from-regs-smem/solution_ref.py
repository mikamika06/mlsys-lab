def predict_occupancy(regs_per_thread, smem_bytes, threads_per_program, limits):
    warp_size = limits["warp_size"]
    warps_per_program = (threads_per_program + warp_size - 1) // warp_size

    blocks_by_regs = limits["register_file"] // (
        regs_per_thread * threads_per_program
    )
    blocks_by_smem = limits["smem_capacity"] // smem_bytes

    resident_blocks = min(
        blocks_by_regs,
        blocks_by_smem,
        limits["max_blocks"],
    )

    return min(
        limits["max_warps"],
        resident_blocks * warps_per_program,
    )
