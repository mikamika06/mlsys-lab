import numpy as np


def _oracle(N, d, bytes_per_element):
    dtype = np.dtype(np.int8)
    scalar_bytes = dtype.itemsize * bytes_per_element
    p = np.empty((N, N), dtype=dtype)
    qkv = np.empty((3, N, d), dtype=dtype)
    return int(p.size * scalar_bytes), int(qkv.size * scalar_bytes)


def grade(sol, fx) -> dict:
    cases = [
        (4096, 64, 2),
        (2048, 128, 2),
        (1024, 64, 4),
        (8192, 128, 2),
    ]

    for N, d, bpe in cases:
        expected = _oracle(N, d, bpe)
        try:
            got = sol.activation_bytes_saved(N, d, bpe)
            got = (int(got[0]), int(got[1]))
        except Exception:
            return {"size_ratio": 0.0}

        if got != expected:
            return {"size_ratio": 0.0}

    return {"size_ratio": 1.0}
