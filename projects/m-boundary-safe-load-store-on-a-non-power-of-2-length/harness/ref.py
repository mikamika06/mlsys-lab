import math
import numpy as np


def ref_vector_add(x, y, n_elements):
    return x[:n_elements] + y[:n_elements]


def ref_wasted_lane_fraction(n_elements, block_size):
    num_blocks = math.ceil(n_elements / block_size)
    total_lanes = num_blocks * block_size
    if total_lanes == 0:
        return 0.0
    return float((total_lanes - n_elements) / total_lanes)


def get_test_cases():
    np.random.seed(42)
    cases = []
    lengths = [1, 13, 97, 255, 300, 1023]
    for n in lengths:
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        cases.append((x, y, n))
    return cases
