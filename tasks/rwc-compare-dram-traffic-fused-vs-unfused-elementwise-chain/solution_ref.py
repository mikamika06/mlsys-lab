def dram_traffic_fused(N, K, dtype_size): return 2 * N * dtype_size
def dram_traffic_unfused(N, K, dtype_size): return 2 * N * (K + 1) * dtype_size
