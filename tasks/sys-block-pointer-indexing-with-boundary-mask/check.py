import numpy as np


def _numpy_oracle(A, row_start, col_start, block_m, block_n):
    out = np.zeros((block_m, block_n), dtype=A.dtype)
    rows = np.arange(row_start, row_start + block_m)
    cols = np.arange(col_start, col_start + block_n)
    row_mask = (rows >= 0) & (rows < A.shape[0])
    col_mask = (cols >= 0) & (cols < A.shape[1])
    if np.any(row_mask) and np.any(col_mask):
        valid_rows = rows[row_mask]
        valid_cols = cols[col_mask]
        out[np.ix_(row_mask, col_mask)] = A[np.ix_(valid_rows, valid_cols)]
    return out


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(12, dtype=np.float32).reshape(3, 4), 1, 2, 3, 3),
        (np.array([[4, -2], [7, 9]], dtype=np.float64), 0, 0, 4, 4),
        (np.arange(20, dtype=np.int32).reshape(4, 5), 3, 4, 2, 3),
        (np.array([[1.5]], dtype=np.float32), 0, 0, 2, 2),
        (np.arange(30, dtype=np.float64).reshape(5, 6), -1, -2, 4, 5),
    ]

    worst = 0.0
    for A, row_start, col_start, block_m, block_n in cases:
        ref = _numpy_oracle(A, row_start, col_start, block_m, block_n)
        try:
            got = sol.block_pointer_gather(
                A, row_start, col_start, block_m, block_n
            )
            got = np.asarray(got)
            if got.shape != ref.shape:
                return {"max_abs_err": float("inf")}
            err = np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))
            worst = max(worst, float(err))
        except Exception:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": worst}
