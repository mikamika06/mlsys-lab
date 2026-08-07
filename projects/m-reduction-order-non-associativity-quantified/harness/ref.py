import numpy as np
from redscale.nonassoc import quantify_non_associativity
from redscale.outliers import detect_loss_spike_ranks
from redscale.overflow import global_overflow_skip

TEST_ARRAYS = [
    np.array([1e-7, 1e7, 1.0, 2.0, -1e7], dtype=np.float32),
    np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    np.array([100.0, -100.0, 50.0, -50.0, 1.0], dtype=np.float32)
]

TEST_MATRICES = [
    np.array([[1.0, 1.0], [1.1, 1.0], [1.0, 100.0]], dtype=np.float32),
    np.array([[2.0, 2.0], [2.2, 2.1], [50.0, 2.0]], dtype=np.float32)
]

TEST_FLAGS = [
    ([False, False, False], False),
    ([True, False, False], True),
    ([False, True, True], True)
]
