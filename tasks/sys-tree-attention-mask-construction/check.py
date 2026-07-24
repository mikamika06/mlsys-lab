import numpy as np


def _oracle(parents):
    n = len(parents)
    mask = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        node = i
        while node != -1:
            mask[i, node] = 1
            node = parents[node]
    return mask


def grade(sol, fx) -> dict:
    cases = [
        [-1],
        [-1, 0, 0],
        [-1, 0, 0, 2, 3],
        [-1, 0, 1, 1, 3, 0, 5],
        [-1, 0, 1, 2, 2, 4, 5, 6],
    ]

    ok = 1.0
    for parents in cases:
        try:
            got = sol.build_tree_attention_mask(list(parents))
        except Exception:
            ok = 0.0
            break

        ref = _oracle(list(parents))
        if not isinstance(got, np.ndarray):
            ok = 0.0
            break
        if got.dtype != np.int64 or got.shape != ref.shape:
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break

    return {"exact_match": ok}
