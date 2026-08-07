import numpy as np
from reference.q4k.quant import quantize_q4_k_superblock, dequantize_q4_k_superblock, round_trip_q4_k
from reference.q4k.analysis import find_worst_subblocks, compare_q4k_q40_mse


def generate_superblock():
    np.random.seed(2026)
    return np.random.uniform(-2.0, 2.0, 256).astype(np.float32)
