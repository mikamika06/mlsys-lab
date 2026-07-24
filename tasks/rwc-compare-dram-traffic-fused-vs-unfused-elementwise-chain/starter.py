import numpy as np

def dram_traffic_fused(N, K, dtype_size):
    return 2 * N  # missing dtype_size factor

def dram_traffic_unfused(N, K, dtype_size):
    return 2 * N * K  # missing +1 and dtype_size factor
