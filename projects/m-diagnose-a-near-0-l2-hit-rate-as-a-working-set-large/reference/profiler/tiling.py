def verify_tiling_dram_reduction(naive_dram_bytes, tiled_dram_bytes, measured_speedup):
    dram_reduced = tiled_dram_bytes < naive_dram_bytes
    ratio = naive_dram_bytes / max(1, tiled_dram_bytes)
    explained = measured_speedup <= (ratio * 1.2) and dram_reduced
    return {
        "dram_bytes_reduced": dram_reduced,
        "dram_reduction_ratio": ratio,
        "speedup_explained_by_dram": explained
    }
