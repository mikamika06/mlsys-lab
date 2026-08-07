import random

random.seed(42)

TEST_CASES_M1 = [
    {"ms": 0.5, "bytes": 1024 * 1024 * 128},
    {"ms": 1.2, "bytes": 1024 * 1024 * 512},
    {"ms": 0.1, "bytes": 1024 * 1024 * 32}
]

TEST_CASES_M2 = [
    {"bytes": 1024 * 1024 * 256, "peak_bw": 1500.0},
    {"bytes": 1024 * 1024 * 1024, "peak_bw": 2000.0}
]

TEST_CASES_M3 = [
    {
        "block_sizes": [16, 32, 64, 128, 256, 512, 1024],
        "gbps_values": [800.0, 1100.0, 1350.0, 1450.0, 1480.0, 1490.0, 1495.0]
    }
]


def compute_gbps(ms, total_bytes):
    return float(total_bytes) / (float(ms) * 1e6)


def roofline_lower_bound_time_ms(total_bytes, peak_bw_gb_s):
    return (float(total_bytes) / (float(peak_bw_gb_s) * 1e9)) * 1e3


def find_roofline_knee(block_sizes, gbps_values):
    import numpy as np
    bs = list(block_sizes)
    gv = list(gbps_values)
    max_val = max(gv)
    threshold = 0.95 * max_val
    for b, v in zip(bs, gv):
        if v >= threshold:
            return b
    return bs[np.argmax(gv)]
