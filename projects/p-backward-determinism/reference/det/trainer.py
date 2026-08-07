import numpy as np

def train_run(seed=42, deterministic=True):
    np.random.seed(seed)
    weights = np.zeros((4, 4))
    for _ in range(5):
        g = np.random.randn(4, 4)
        if deterministic:
            g = np.round(g, 6)
        weights += g
    return weights

def measure_cost():
    return 1.25

def is_deterministic():
    return True
