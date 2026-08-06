import numpy as np

def synthetic_exploding_gradients():
    rng = np.random.RandomState(42)
    steps = 10
    base = rng.randn(steps, 4) * 10.0
    scales = np.exp(np.linspace(0, 5, steps))
    return [b * s for b, s in zip(base, scales)]

def expected_threshold(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        return float(max_norm / total_norm)
    return 1.0
