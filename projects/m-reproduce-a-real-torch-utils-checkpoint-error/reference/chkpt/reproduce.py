import numpy as np


def trigger_error():
    np.random.seed(42)
    w1 = np.random.randn(32, 32)
    w2 = np.random.randn(32, 32)[:-1, :]
    x = np.random.randn(10, 32)
    try:
        h = np.dot(x, w1)
        _ = np.dot(h, w2)
        return False
    except Exception:
        return True
