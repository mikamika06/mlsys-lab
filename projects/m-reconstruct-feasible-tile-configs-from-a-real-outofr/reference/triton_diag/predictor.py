def verify_out_of_resources(block_m: int, block_n: int, num_stages: int, threads_per_warp: int = 32) -> bool:
    smem = block_m * block_n * 4 * num_stages
    max_smem = 49152
    return smem <= max_smem
