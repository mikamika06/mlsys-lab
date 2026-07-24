import numpy as np


def _oracle(W):
    W = np.asarray(W)
    rows, cols = W.shape
    out = np.zeros((rows, cols), dtype=np.int64)
    for r in range(rows):
        for start in range(0, cols, 4):
            group = W[r, start:start + 4]
            order = sorted(range(4), key=lambda i: (-abs(float(group[i])), i))
            for idx in order[:2]:
                out[r, start + idx] = 1
    return out


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.0, -5.0, 2.0, 0.5, 9.0, 1.0, 0.0, -2.0]]),
        np.array([
            [8.0, 1.0, -7.0, 2.0, 0.1, 0.2, 0.3, 10.0],
            [-4.0, -3.0, 2.0, 1.0, 7.0, -9.0, 6.0, 5.0],
        ]),
        np.array([
            [0.5, -0.5, 0.4, 0.3, -8.0, 1.0, 2.0, 3.0],
            [11.0, 10.0, 9.0, 8.0, 1.0, -20.0, 2.0, 3.0],
        ]),
    ]

    ok = 1.0
    for W in cases:
        ref = _oracle(W)
        try:
            got = np.asarray(sol.select_2_4_mask(W))
        except Exception:
            ok = 0.0
            break
        if got.shape != ref.shape or not np.array_equal(got, ref):
            ok = 0.0
            break
        for r in range(got.shape[0]):
            for start in range(0, got.shape[1], 4):
                if int(np.sum(got[r, start:start + 4])) != 2:
                    ok = 0.0
                    break
    return {"exact_match": ok}
