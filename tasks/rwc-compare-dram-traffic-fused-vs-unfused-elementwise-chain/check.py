import numpy as np

def _ref_dram_traffic_fused(N, K, dtype_size):
    return 2 * N * dtype_size

def _ref_dram_traffic_unfused(N, K, dtype_size):
    return 2 * N * (K + 1) * dtype_size

def grade(sol, fx) -> dict:
    cases = [
        (1000, 10, 8),
        (10000, 5, 4),
        (500, 20, 2),
    ]
    ok = 1.0
    for N, K, dtype_size in cases:
        try:
            fused_traffic = sol.dram_traffic_fused(N, K, dtype_size)
            unfused_traffic = sol.dram_traffic_unfused(N, K, dtype_size)
        except Exception:
            ok = 0.0
            break
        if fused_traffic != _ref_dram_traffic_fused(N, K, dtype_size) or unfused_traffic != _ref_dram_traffic_unfused(N, K, dtype_size):
            ok = 0.0
            break
    return {"exact_match": ok}
