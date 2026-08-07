import sys
import numpy as np

sys.path.insert(0, ".")
from gradacc.accumulate import accumulate, full_batch
from gradacc.memory import peak_memory_sweep


def test_accumulated_matches_full():
    np.random.seed(42)
    W = np.random.randn(10, 10)
    b = np.zeros(10)
    batches = [(np.random.randn(5, 10), np.random.randn(5, 10)) for _ in range(4)]

    X_full = np.concatenate([batch[0] for batch in batches], axis=0)
    Y_full = np.concatenate([batch[1] for batch in batches], axis=0)

    dW_acc, db_acc = accumulate(batches, W, b, steps=4)
    dW_full, db_full = full_batch(X_full, Y_full, W, b)

    np.testing.assert_allclose(dW_acc, dW_full, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(db_acc, db_full, rtol=1e-5, atol=1e-5)


def test_memory_flat():
    W = np.zeros((100, 100))
    b = np.zeros(100)

    def gen(steps):
        return [(np.ones((200, 100)), np.ones((200, 100))) for _ in range(steps)]

    peaks = peak_memory_sweep(gen, W, b, [1, 2, 4])
    assert len(peaks) == 3
    assert peaks[-1] < peaks[0] * 1.5
