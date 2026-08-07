import numpy as np

np.random.seed(42)
TENSORS = [np.random.randn(32, 32).astype(np.float32) for _ in range(5)]
