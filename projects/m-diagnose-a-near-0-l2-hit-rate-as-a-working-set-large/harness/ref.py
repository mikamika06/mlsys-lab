import numpy as np


def generate_workload_cases():
    np.random.seed(42)
    cases = []
    for l2_size in [1024 * 1024, 2 * 1024 * 1024]:
        for working_set_size in [512 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024]:
            hit_rate = 0.95 if working_set_size <= l2_size else 0.005
            cases.append({
                "l2_capacity_bytes": l2_size,
                "working_set_bytes": working_set_size,
                "measured_l2_hit_rate": hit_rate
            })
    return cases


def generate_tiling_cases():
    np.random.seed(1337)
    cases = []
    for m, n, k in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        tile_m, tile_n, tile_k = 32, 32, 32
        naive_bytes = 2 * (m * k + k * n + m * n) * 4
        tiled_bytes = 2 * (m * k * (k // tile_k) + k * n * (n // tile_n)) * 4
        tiled_bytes = int(naive_bytes * 0.15)
        speedup = float(naive_bytes / max(1, tiled_bytes))
        cases.append({
            "m": m,
            "n": n,
            "k": k,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "naive_dram_bytes": naive_bytes,
            "tiled_dram_bytes": tiled_bytes,
            "measured_speedup": min(5.0, max(1.1, speedup * 0.3))
        })
    return cases
