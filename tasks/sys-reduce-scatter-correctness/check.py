import numpy as np


def _oracle(chunks):
    ranks = len(chunks)
    return [
        np.sum(np.stack([chunks[r][i] for r in range(ranks)]), axis=0)
        for i in range(ranks)
    ]


def grade(sol, fx) -> dict:
    cases = [
        [
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
        ],
        [
            [
                np.array([0.5, -2.0, 3.0]),
                np.array([1.0, 4.0, -1.0]),
                np.array([2.5, 0.0, 8.0]),
            ],
            [
                np.array([-1.5, 1.0, 2.0]),
                np.array([3.0, -4.0, 5.0]),
                np.array([1.0, 2.0, -3.0]),
            ],
            [
                np.array([4.0, 2.0, -5.0]),
                np.array([0.5, 1.5, 1.0]),
                np.array([-2.0, 3.0, 6.0]),
            ],
        ],
        [
            [
                np.array([10.0]),
                np.array([-1.0]),
                np.array([2.0]),
                np.array([3.0]),
            ],
            [
                np.array([-5.0]),
                np.array([4.0]),
                np.array([8.0]),
                np.array([1.0]),
            ],
            [
                np.array([2.0]),
                np.array([0.0]),
                np.array([-3.0]),
                np.array([7.0]),
            ],
            [
                np.array([1.0]),
                np.array([5.0]),
                np.array([6.0]),
                np.array([-2.0]),
            ],
        ],
    ]

    max_err = 0.0
    for chunks in cases:
        try:
            got = sol.reduce_scatter_sum(chunks)
            ref = _oracle(chunks)
            for a, b in zip(got, ref):
                max_err = max(max_err, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        except Exception:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": max_err}
