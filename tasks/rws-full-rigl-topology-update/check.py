import numpy as np


def _oracle(w, grad, mask, update_fraction):
    out = np.asarray(mask, dtype=np.int64).copy()
    live = int(np.sum(out))
    k = int(np.floor(update_fraction * live))

    active = np.flatnonzero(out == 1)
    inactive = np.flatnonzero(out == 0)

    if k:
        drop_order = sorted(active.tolist(), key=lambda i: (abs(float(w[i])), i))
        dropped = drop_order[:k]
        out[dropped] = 0

        candidates = np.flatnonzero(out == 0)
        grow_order = sorted(
            candidates.tolist(),
            key=lambda i: (-abs(float(grad[i])), i),
        )
        grown = grow_order[:k]
        out[grown] = 1

    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.1, 5.0, 0.2, 0.0]),
            np.array([1.0, 0.0, 3.0, 2.0]),
            np.array([1, 1, 0, 0]),
            0.5,
        ),
        (
            np.array([4.0, -1.0, 1.0, 2.0, 0.5]),
            np.array([0.2, 8.0, 7.0, 1.0, 6.0]),
            np.array([1, 1, 0, 1, 0]),
            0.67,
        ),
        (
            np.array([1.0, 1.0, 3.0, 3.0, 2.0]),
            np.array([5.0, 5.0, 1.0, 9.0, 9.0]),
            np.array([1, 1, 0, 0, 0]),
            0.5,
        ),
        (
            np.array([0.0, -8.0, 2.0, -2.0, 7.0, 1.0]),
            np.array([10.0, 1.0, 1.0, 4.0, 3.0, 2.0]),
            np.array([1, 0, 1, 0, 1, 0]),
            0.34,
        ),
    ]

    ok = 1.0
    for w, g, m, frac in cases:
        try:
            got = np.asarray(
                sol.rigl_topology_update(w, g, m, frac),
                dtype=np.int64,
            )
        except Exception:
            ok = 0.0
            break

        expected = _oracle(w, g, m, frac)
        if not np.array_equal(got, expected):
            ok = 0.0
            break
        if int(np.sum(got)) != int(np.sum(m)):
            ok = 0.0
            break

    return {"exact_match": ok}
