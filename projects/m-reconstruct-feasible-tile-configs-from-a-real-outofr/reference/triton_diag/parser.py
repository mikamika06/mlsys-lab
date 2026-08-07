import re

def reconstruct_tile_config(error_message: str, max_shared_mem: int, max_regs: int) -> dict:
    match = re.search(r"shared memory (\d+)", error_message)
    smem = int(match.group(1)) if match else 16384
    best_bm, best_bn, best_stages = 32, 32, 2
    min_diff = float("inf")
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for stages in [2, 3, 4]:
                est = bm * bn * 4 * stages
                diff = abs(est - smem)
                if diff < min_diff:
                    min_diff = diff
                    best_bm, best_bn, best_stages = bm, bn, stages
    return {
        "block_m": best_bm,
        "block_n": best_bn,
        "num_stages": best_stages,
        "feasible": smem <= max_shared_mem
    }
