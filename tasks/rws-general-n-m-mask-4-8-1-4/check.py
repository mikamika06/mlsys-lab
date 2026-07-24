import numpy as np


def _oracle_mask(weights, N, M):
    weights = np.asarray(weights)
    mask = np.zeros(weights.shape, dtype=np.int8)
    for start in range(0, len(weights), M):
        group = weights[start:start + M]
        order = np.argsort(-np.abs(group), kind="stable")
        keep = order[:N]
        mask[start + keep] = 1
    return mask


def grade(sol, fx) -> dict:
    cases = [
        (np.array([0.1, -3.0, 2.0, 0.5, 8.0, -1.0, 4.0, 2.0]), 2, 4),
        (np.array([1.0, -1.0, 1.0, 0.2, -5.0, 3.0, 2.0, 2.0, 0.0, -4.0, 7.0, 1.0]), 1, 4),
        (np.array([0.4, -9.0, 8.0, -7.0, 6.0, 5.0, -4.0, 3.0]), 3, 4),
        (np.linspace(-2.5, 2.5, 24), 1, 4),
        (np.array([0.0, -0.0, 2.0, -2.0, 3.0, -3.0, 1.0, -1.0]), 2, 4),
        (np.arange(-16, 16, dtype=np.float64), 4, 8),
    ]

    exact = 1.0
    for weights, N, M in cases:
        try:
            got = np.asarray(sol.nm_mask(weights.copy(), N, M))
        except Exception:
            exact = 0.0
            break

        ref = _oracle_mask(weights, N, M)

        if got.shape != ref.shape or not np.array_equal(got, ref):
            exact = 0.0
            break

        for start in range(0, len(weights), M):
            if int(np.sum(got[start:start + M])) != N:
                exact = 0.0
                break
        if exact == 0.0:
            break

        dropped_got = np.sum(np.abs(weights[got == 0]))
        dropped_ref = np.sum(np.abs(weights[ref == 0]))
        if not np.isclose(dropped_got, dropped_ref, rtol=0.0, atol=0.0):
            exact = 0.0
            break

    return {"exact_match": exact}
