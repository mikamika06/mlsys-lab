import numpy as np


def get_test_cases():
    np.random.seed(42)
    cases = []
    for _ in range(5):
        layers = []
        for _ in range(2):
            k = np.random.randn(1, 4, 16, 32).astype(np.float32)
            v = np.random.randn(1, 4, 16, 32).astype(np.float32)
            layers.append((k, v))
        noise = np.random.uniform(-1e-6, 1e-6, size=(1, 4, 16, 32)).astype(np.float32)
        cand_layers = [(k + noise, v + noise) for k, v in layers]
        cases.append((layers, cand_layers, 1e-5))
    return cases
