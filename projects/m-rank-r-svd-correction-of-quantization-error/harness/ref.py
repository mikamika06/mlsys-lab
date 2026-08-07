import numpy as np

np.random.seed(42)

TEST_WEIGHTS = [
    np.random.randn(64, 64),
    np.random.randn(32, 64),
    np.random.randn(48, 48)
]

HINVS = [np.eye(w.shape[0]) for w in TEST_WEIGHTS]
ROT_MATRICES = [
    np.linalg.qr(np.random.randn(w.shape[0], w.shape[0]))[0]
    for w in TEST_WEIGHTS
]
