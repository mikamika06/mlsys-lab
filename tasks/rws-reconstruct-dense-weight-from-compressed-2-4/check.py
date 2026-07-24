import numpy as np


def _oracle(values, indices, shape):
    rows, cols = shape
    out = np.zeros(shape, dtype=np.float32)
    groups = cols // 4
    for r in range(rows):
        for g in range(groups):
            base = g * 4
            for j in range(2):
                pos = base + int(indices[r, g * 2 + j])
                out[r, pos] = np.float32(values[r, g * 2 + j])
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.5, 2.5, 3.0, 4.0]], dtype=np.float32),
            np.array([[0, 3, 1, 2]], dtype=np.int8),
            (1, 8),
        ),
        (
            np.array(
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                dtype=np.float32,
            ),
            np.array(
                [[0, 2, 1, 3, 0, 1], [3, 1, 2, 0, 1, 3]],
                dtype=np.int8,
            ),
            (2, 12),
        ),
        (
            np.arange(16, dtype=np.float32).reshape(1, 16)[:, ::2],
            np.array([[0, 1, 2, 3, 1, 0, 3, 2]], dtype=np.int8),
            (1, 16),
        ),
    ]
    ok = 1.0
    for values, indices, shape in cases:
        try:
            got = sol.reconstruct_24(values, indices, shape)
        except Exception:
            ok = 0.0
            break
        ref = _oracle(values, indices, shape)
        if not isinstance(got, np.ndarray):
            ok = 0.0
            break
        if got.dtype != np.float32 or not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
