import numpy as np


def _oracle(prefix_lengths, chunk_size):
    p = np.asarray(prefix_lengths, dtype=np.float64)
    c = float(chunk_size)
    reused = np.floor(p / c) * c
    return float(np.mean(reused / p))


def grade(sol, fx) -> dict:
    cases = [
        (np.array([10, 11, 15], dtype=np.int64), 4),
        (np.array([1, 2, 3, 4, 5, 16, 31], dtype=np.int64), 8),
        (np.array([64, 128, 256, 512], dtype=np.int64), 64),
        (np.array([7, 13, 29, 31, 100, 101], dtype=np.int64), 16),
        (np.arange(1, 100, dtype=np.int64), 7),
    ]
    max_err = 0.0
    for prefixes, chunk in cases:
        try:
            got = float(sol.prefix_chunk_hit_rate(prefixes, chunk))
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(prefixes, chunk)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        max_err = max(max_err, err)
    return {"rel_err": max_err}
