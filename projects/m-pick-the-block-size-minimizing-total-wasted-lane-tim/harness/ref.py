import numpy as np

CANDIDATES = [16, 32, 64, 128, 256, 512, 1024]

TEST_WORKLOADS = [
    {"n": 100, "launch_overhead": 5.0, "candidates": CANDIDATES},
    {"n": 1024, "launch_overhead": 12.0, "candidates": CANDIDATES},
    {"n": 1025, "launch_overhead": 8.0, "candidates": CANDIDATES},
    {"n": 3000, "launch_overhead": 20.0, "candidates": CANDIDATES},
    {"n": 4096, "launch_overhead": 15.0, "candidates": CANDIDATES},
    {"n": 5000, "launch_overhead": 10.0, "candidates": CANDIDATES},
    {"n": 7777, "launch_overhead": 25.0, "candidates": CANDIDATES},
    {"n": 12345, "launch_overhead": 50.0, "candidates": CANDIDATES},
    {"n": 65536, "launch_overhead": 100.0, "candidates": CANDIDATES},
    {"n": 100000, "launch_overhead": 2.0, "candidates": CANDIDATES},
]


def calculate_wasted_lane_time(n, block_size, launch_overhead):
    num_blocks = (n + block_size - 1) // block_size
    remainder = n % block_size
    padding_lanes = (block_size - remainder) if remainder != 0 else 0
    tail_waste = padding_lanes
    fixed_overhead = num_blocks * launch_overhead
    return float(tail_waste + fixed_overhead)


def select_best_block_size(n, candidates, launch_overhead):
    wastes = [
        calculate_wasted_lane_time(n, b, launch_overhead) for b in candidates
    ]
    best_idx = int(np.argmin(wastes))
    return best_idx, wastes[best_idx]
