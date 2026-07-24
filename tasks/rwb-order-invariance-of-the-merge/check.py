import numpy as np


def _oracle_merge(partials):
    arr = np.asarray(partials, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    order = np.argsort(np.arange(len(arr)))
    ordered = arr[order]
    return np.sum(ordered, axis=0, dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = [
        [
            np.array([1e16, 2.0, -3.0]),
            np.array([-1e16, 4.0, 5.0]),
            np.array([7.0, -2.0, 1.0]),
        ],
        [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([-0.5, -0.7, 1.2]),
            np.array([3.0, -2.0, 8.0]),
        ],
        [
            np.array([1e-8, 1e8, 1.0]),
            np.array([-1e-8, -1e8, 2.0]),
            np.array([5.0, 6.0, -3.0]),
            np.array([4.0, 7.0, 9.0]),
        ],
    ]

    max_err = 0.0
    max_order_err = 0.0

    for partials in cases:
        try:
            got_a, got_b = sol.merge_partials([x.copy() for x in partials])
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "order_max_abs_err": float("inf"),
            }

        ref = _oracle_merge(partials)
        got_a = np.asarray(got_a, dtype=np.float64)
        got_b = np.asarray(got_b, dtype=np.float64)

        if got_a.shape != ref.shape or got_b.shape != ref.shape:
            return {
                "max_abs_err": float("inf"),
                "order_max_abs_err": float("inf"),
            }

        max_err = max(
            max_err,
            float(np.max(np.abs(got_a - ref))),
            float(np.max(np.abs(got_b - ref))),
        )
        max_order_err = max(max_order_err, float(np.max(np.abs(got_a - got_b))))

    return {
        "max_abs_err": max_err,
        "order_max_abs_err": max_order_err,
    }
