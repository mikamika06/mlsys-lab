import numpy as np


def _oracle(parents):
    n = len(parents)
    out = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        cur = i
        while cur != -1:
            out[i, cur] = 1
            cur = parents[cur]
    return out


def grade(sol, fx) -> dict:
    cases = [
        [-1, 0],
        [-1, 0, 0, 1, 1, 2],
        [-1, 0, 1, 2, 0, 4, 4],
        [-1, 0, 0, 0, 3, 3, 1, 6],
        [-1, 0, 1, 1, 2, 2, 5, 0, 7],
    ]

    ok = 1.0
    for parents in cases:
        try:
            got = sol.build_tree_mask(list(parents))
            got = np.asarray(got)
        except Exception:
            ok = 0.0
            break

        ref = _oracle(list(parents))
        if got.dtype != np.int8:
            ok = 0.0
            break
        if got.shape != ref.shape or not np.array_equal(got, ref):
            ok = 0.0
            break

    return {"exact_match": ok}
