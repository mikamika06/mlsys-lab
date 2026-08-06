import numpy as np


def generate_test_cases():
    np.random.seed(1337)
    cases = []
    for i in range(5):
        out_dim = 16 * (i + 1)
        in_dim = 32 * (i + 1)
        seq_len = 64
        W = np.random.randn(out_dim, in_dim)
        X = np.random.randn(in_dim, seq_len)
        sparsity = 0.25 + (i * 0.1)
        num_bits = 4 if i % 2 == 0 else 8
        cases.append((W, X, sparsity, num_bits))
    return cases
