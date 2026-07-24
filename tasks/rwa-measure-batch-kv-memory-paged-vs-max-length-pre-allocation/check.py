import numpy as np


def _oracle(lengths, max_len, block_size):
    arr = np.asarray(lengths, dtype=np.int64)
    paged = np.sum(((arr + block_size - 1) // block_size) * block_size)
    contiguous = arr.shape[0] * max_len
    return float(contiguous / paged)


def grade(sol, fx) -> dict:
    cases = [
        ([10, 33, 70], 100, 16),
        ([1, 2, 3, 4], 128, 8),
        ([16, 32, 48], 64, 16),
        ([17, 31, 63, 129], 256, 32),
        (list(range(1, 51)), 512, 64),
    ]

    worst = 0.0
    for lengths, max_len, block_size in cases:
        try:
            got = float(sol.kv_memory_waste_ratio(list(lengths), max_len, block_size))
        except Exception:
            return {"size_ratio": 0.0}

        ref = _oracle(lengths, max_len, block_size)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"size_ratio": 1.0 if worst < 1e-9 else 0.0}
