import numpy as np


def generate_analytical_cases():
    np.random.seed(42)
    cases = []
    for _ in range(10):
        seq_len = int(np.random.choice([2048, 4096, 8192, 16384]))
        hidden_dim = int(np.random.choice([1024, 2048, 4096]))
        world_size = int(np.random.choice([2, 4, 8, 16]))
        dtype_bytes = 2
        cases.append({
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "world_size": world_size,
            "dtype_bytes": dtype_bytes,
        })
    return cases
