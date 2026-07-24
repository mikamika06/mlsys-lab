"""Generate softmax_overflow.npy fixture with logits up to +1000."""
import numpy as np

def gen():
    rng = np.random.default_rng(42)
    row1 = np.array([1000.0, 999.0, 998.0, 997.0, 0.0, -10.0, -50.0])
    row2 = np.full(7, 1000.0)
    row3 = np.full(7, -1000.0)
    row4 = np.array([1000.0, -1000.0, 500.0, -500.0, 0.0, 200.0, -200.0])
    batch = rng.uniform(-1000, 1000, size=(20, 7))
    all_rows = np.vstack([row1, row2, row3, row4, batch])
    np.save("softmax_overflow.npy", all_rows)

if __name__ == "__main__":
    gen()
