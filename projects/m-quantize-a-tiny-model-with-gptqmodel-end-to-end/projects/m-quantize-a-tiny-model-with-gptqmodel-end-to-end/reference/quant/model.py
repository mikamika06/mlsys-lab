import numpy as np


def create_tiny_model():
    np.random.seed(42)
    return {
        "weight": np.random.randn(64, 64).astype(np.float32),
        "bias": np.zeros((64,), dtype=np.float32),
    }
