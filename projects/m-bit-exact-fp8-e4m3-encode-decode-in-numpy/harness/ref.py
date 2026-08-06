import numpy as np


def generate_kv_dumps():
    rng = np.random.default_rng(2026)
    dumps = []

    d1 = rng.normal(loc=0.0, scale=2.5, size=(4, 12, 64, 64)).astype(np.float32)
    dumps.append(d1)

    d2 = rng.uniform(low=-100.0, high=100.0, size=(2, 8, 32, 128)).astype(np.float32)
    dumps.append(d2)

    d3 = np.zeros((2, 4, 16, 16), dtype=np.float32)
    d3[0, 0, 0, 0] = 0.001
    dumps.append(d3)

    return dumps


def generate_e4m3_test_cases():
    all_bytes = np.arange(256, dtype=np.uint8)
    return all_bytes
