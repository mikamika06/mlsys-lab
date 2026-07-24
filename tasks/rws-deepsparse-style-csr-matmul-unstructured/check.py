import numpy as np


def _dense_from_csr(data, indices, indptr, shape):
    dense = np.zeros(shape, dtype=np.float64)
    rows = shape[0]
    for r in range(rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for p in range(start, end):
            dense[r, int(indices[p])] = float(data[p])
    return dense


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.5, -2.0, 3.0, 4.5]),
            np.array([0, 3, 1, 2]),
            np.array([0, 2, 3, 4]),
            np.array([[2.0, -1.0], [0.5, 3.0], [4.0, 1.0], [-2.0, 5.0]]),
        ),
        (
            np.array([0.25, -1.75, 2.25, 5.0, -3.0]),
            np.array([4, 0, 3, 1, 5]),
            np.array([0, 1, 3, 5]),
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0],
                ]
            ),
        ),
        (
            np.array([3.0, -4.0]),
            np.array([2, 0]),
            np.array([0, 1, 2, 2, 2]),
            np.array([[1.0], [2.0], [3.0]]),
        ),
    ]

    max_err = 0.0
    for data, indices, indptr, X in cases:
        try:
            got = np.asarray(sol.csr_matmul(data, indices, indptr, X), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        shape = (len(indptr) - 1, X.shape[0])
        dense = _dense_from_csr(data, indices, indptr, shape)
        ref = dense @ X
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
