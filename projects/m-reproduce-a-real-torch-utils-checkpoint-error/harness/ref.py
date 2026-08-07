import numpy as np


class MockModule:
    def __init__(self, size=16, seed=42):
        self.size = size
        np.random.seed(seed)
        self.weights = np.random.randn(size, size)

    def forward_normal(self, x):
        return np.tanh(np.dot(x, self.weights))

    def forward_broken(self, x, step):
        if step % 2 == 1:
            return np.tanh(np.dot(x, self.weights[:-1, :]))
        return np.tanh(np.dot(x, self.weights))


def simulate_error():
    mod = MockModule(size=32)
    x = np.random.randn(10, 32)
    try:
        for step in range(5):
            if step == 3:
                _ = mod.forward_broken(x, step)
            else:
                _ = mod.forward_normal(x)
        return False
    except Exception:
        return True


def simulate_profile(n_layers, checkpoint_every):
    total_mem = n_layers * 100.0
    if checkpoint_every > 0:
        saved_mem = total_mem * (1.0 / checkpoint_every) + (n_layers * 5.0)
        time_cost = n_layers * 1.0 + (n_layers / checkpoint_every) * 0.2
    else:
        saved_mem = total_mem
        time_cost = n_layers * 1.0
    return float(saved_mem), float(time_cost)


def compute_optimal_interval(n_layers, memory_limit):
    if n_layers <= 0:
        return 1
    val = np.sqrt(n_layers / max(1.0, memory_limit))
    opt = int(np.ceil(val))
    return max(1, min(opt, n_layers))
