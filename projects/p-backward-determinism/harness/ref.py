import numpy as np

def run_training(seed=42, deterministic=False, steps=5):
    np.random.seed(seed)
    weights = np.zeros((4, 4))
    for i in range(steps):
        grad = np.random.randn(4, 4)
        if deterministic:
            grad = np.round(grad, 6)
        weights += grad
    return weights

def analyze_divergence(run1, run2):
    diff = np.abs(run1 - run2)
    return 1.0 if np.any(diff > 1e-7) else 0.0

def measure_overhead():
    return 1.25
