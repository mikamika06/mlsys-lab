import numpy as np

def generate_test_data(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
    outliers = rng.choice([0.1, 1.0, 10.0, 100.0], size=(16, 16), p=[0.7, 0.2, 0.08, 0.02])
    data = x * outliers
    return data
