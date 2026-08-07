import numpy as np


def generate_test_cases():
    cases = []
    np.random.seed(42)
    for _ in range(5):
        lse = np.random.randn(4, 8)
        denom = np.exp(lse)
        cases.append((lse, denom))
    return cases
