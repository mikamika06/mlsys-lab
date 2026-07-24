import numpy as np


def _oracle(x, block_size):
    n = len(x)
    blocks = (n + block_size - 1) // block_size
    out = np.zeros(blocks, dtype=np.float64)
    for k in range(blocks):
        start = k * block_size
        end = min(start + block_size, n)
        out[k] = np.sum(x[start:end], dtype=np.float64)
    return out


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(17, dtype=np.float32) * 0.5 + 1.0, 8),
        (np.array([1.25, -2.0, 3.5, 4.0, 9.0], dtype=np.float64), 4),
        (np.linspace(-3, 7, 31, dtype=np.float32), 16),
        (np.array([2.0, 5.0, -1.0], dtype=np.float64), 5),
    ]

    worst = 0.0
    for x, block_size in cases:
        ref = _oracle(x, block_size)
        try:
            got = sol.masked_block_sum(x, block_size)
        except Exception:
            return {"rel_err": float("inf")}
        err = _rel_err(got, ref)
        worst = max(worst, err)
    return {"rel_err": worst}
