import numpy as np


def generate_test_deltas(seed=1337, count=100, shape=(64, 64)):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) * 0.01 for _ in range(count)]


def generate_loss_series(seed=2026):
    rng = np.random.RandomState(seed)
    base = np.linspace(2.5, 0.5, 50)
    noise = rng.normal(0, 0.01, 50)
    series = (base + noise).tolist()
    series[25] *= 3.5
    return series
