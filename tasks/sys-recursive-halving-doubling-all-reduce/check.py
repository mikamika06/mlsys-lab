import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        [
            np.array([1.5, -2.0, 3.25]),
            np.array([4.0, 5.5, -1.25]),
        ],
        [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 7.0, 8.0]),
            np.array([-1.0, 2.5, 0.5, 3.5]),
            np.array([9.0, -4.0, 1.5, -2.0]),
        ],
        [
            np.linspace(-1.0, 1.0, 8),
            np.linspace(2.0, 3.0, 8),
            np.linspace(4.0, 5.0, 8),
            np.linspace(-3.0, -2.0, 8),
            np.linspace(7.0, 8.0, 8),
            np.linspace(1.0, 2.0, 8),
            np.linspace(-5.0, -4.0, 8),
            np.linspace(0.0, 1.0, 8),
        ],
    ]

    worst = 0.0
    for buffers in cases:
        try:
            got = sol.recursive_halving_doubling_all_reduce(
                [x.copy() for x in buffers]
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if len(got) != len(buffers):
            return {"max_abs_err": float("inf")}

        # NumPy is the numeric oracle for the all-reduce sum.
        ref = np.sum(np.stack(buffers, axis=0), axis=0)

        for value in got:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != ref.shape:
                return {"max_abs_err": float("inf")}
            worst = max(worst, float(np.max(np.abs(arr - ref))))

    return {"max_abs_err": worst}
