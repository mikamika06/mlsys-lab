import numpy as np


def get_m1_fixtures():
    rng = np.random.default_rng(123)
    cases = []
    for _ in range(5):
        bw = rng.normal(size=(32, 32))
        aa = rng.normal(size=(32, 8))
        ab = rng.normal(size=(8, 32))
        x = rng.normal(size=(4, 32))
        scaling = float(rng.uniform(0.5, 2.0))
        cases.append((bw, aa, ab, scaling, x))
    return cases


def get_m2_fixtures():
    return [
        {"vram_limit_mb": 8192, "base_mb": 2048, "rank": 16, "seq_len": 512, "target_tokens": 65536},
        {"vram_limit_mb": 4096, "base_mb": 1536, "rank": 8, "seq_len": 256, "target_tokens": 32768}
    ]


def get_m3_fixtures():
    return [
        ({"r": 8, "alpha": 16}, {"r": 16, "alpha": 32}),
        ({"target": "q"}, {"target": "v"})
    ]
