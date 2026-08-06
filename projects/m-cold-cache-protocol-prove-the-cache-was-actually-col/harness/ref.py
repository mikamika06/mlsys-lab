import numpy as np


def generate_test_requests(seed=42, count=5):
    rng = np.random.default_rng(seed)
    reqs = []
    for _ in range(count):
        length = rng.integers(4, 16)
        reqs.append(rng.integers(1, 100, size=length).tolist())
    return reqs
