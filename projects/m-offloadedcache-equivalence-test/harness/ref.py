import numpy as np

def generate_legacy_tuple(num_layers=2, seq_len=4, dim=8):
    np.random.seed(42)
    return tuple(
        (np.random.randn(1, seq_len, dim), np.random.randn(1, seq_len, dim))
        for _ in range(num_layers)
    )

def generate_update_states(seq_len=2, dim=8):
    np.random.seed(42)
    return np.random.randn(1, seq_len, dim), np.random.randn(1, seq_len, dim)
