import numpy as np


def _oracle_pack_int4(values):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    for i in range(0, values.size, 2):
        low = int(values[i]) & 15
        high = int(values[i + 1]) & 15 if i + 1 < values.size else 0
        out[i // 2] = np.uint8(low | (high << 4))
    return out


def _byte_exact_fraction(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.mean(a.tobytes() == b.tobytes())) if False else (
        sum(x == y for x, y in zip(a.tobytes(), b.tobytes())) / len(a.tobytes())
    )


def grade(sol, fx) -> dict:
    cases = [
        np.array([0, 1], dtype=np.int64),
        np.array([1, 2, 15, 0], dtype=np.int64),
        np.array([15, 15, 0, 0, 7], dtype=np.int64),
        np.arange(16, dtype=np.int64),
        np.array([3, 8, 12, 4, 1, 14, 9], dtype=np.int64),
    ]
    ok = 1.0
    for values in cases:
        expected = _oracle_pack_int4(values)
        try:
            got = sol.pack_int4(values)
            got = np.asarray(got, dtype=np.uint8)
        except Exception:
            ok = 0.0
            break
        if _byte_exact_fraction(expected, got) != 1.0:
            ok = 0.0
            break
    return {"byte_exact_fraction": ok}
