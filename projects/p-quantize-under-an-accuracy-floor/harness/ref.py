import numpy as np


class MockModel:
    """Mock neural network model for testing quantization routines."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.layers = {
            "l1_sensitive": np.random.randn(16, 16).astype(np.float32) * 3.0,
            "l2_robust": np.random.randn(16, 16).astype(np.float32) * 0.2,
            "l3_robust": np.random.randn(16, 16).astype(np.float32) * 0.1,
            "l4_robust": np.random.randn(16, 16).astype(np.float32) * 0.15,
        }

    def forward(self, x):
        h = x
        for w in self.layers.values():
            h = np.dot(h, w)
        return h


def get_dataset(seed=42, n=20):
    """Generates synthetic dataset samples for evaluation."""
    np.random.seed(seed)
    ds = []
    for _ in range(n):
        x = np.random.randn(1, 16).astype(np.float32)
        target = int(np.argmax(x))
        ds.append((x, target))
    return ds


def get_calib_data(seed=123, n=10):
    """Generates calibration activation vectors."""
    np.random.seed(seed)
    return [np.random.randn(1, 16).astype(np.float32) for _ in range(n)]
