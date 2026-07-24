import numpy as np


def _oracle_compress(W):
    m, n = W.shape
    groups = n // 4
    mask = np.zeros((m, n), dtype=np.int64)
    values = np.zeros((m, groups * 2), dtype=np.float64)
    indices = np.zeros((m, groups * 2), dtype=np.uint8)

    for i in range(m):
        for g in range(groups):
            grp = W[i, g * 4:(g + 1) * 4]
            order = np.argsort(-np.abs(grp), kind="stable")  # descending |value|, ties -> lower index first
            keep = np.sort(order[:2])
            mask[i, g * 4 + keep] = 1
            values[i, g * 2:(g + 1) * 2] = grp[keep]
            indices[i, g * 2:(g + 1) * 2] = keep.astype(np.uint8)

    return mask, values, indices


def _oracle_output(W, X):
    mask, _, _ = _oracle_compress(W)
    pruned = W * mask
    return pruned @ X


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(9)
    mask_ok = 1.0
    worst_err = 0.0

    for _ in range(5):
        m = int(rng.integers(2, 8))
        groups = int(rng.integers(1, 6))
        n = groups * 4
        p = int(rng.integers(2, 6))

        W = rng.standard_normal((m, n))
        X = rng.standard_normal((n, p))

        ref_mask, _, _ = _oracle_compress(W)
        ref_output = _oracle_output(W, X)

        try:
            got = sol.prune24_compress_and_matmul(W.copy(), X.copy())
            got_mask, _got_values, _got_indices, got_output = got
            got_mask = np.asarray(got_mask)
            got_output = np.asarray(got_output, dtype=np.float64)
        except Exception:
            return {"mask_exact": 0.0, "max_abs_err": float("inf")}

        if got_mask.shape != ref_mask.shape or not np.array_equal(got_mask.astype(np.int64), ref_mask):
            mask_ok = 0.0

        if got_output.shape != ref_output.shape:
            return {"mask_exact": mask_ok, "max_abs_err": float("inf")}

        worst_err = max(worst_err, float(np.max(np.abs(got_output - ref_output))))

    return {"mask_exact": mask_ok, "max_abs_err": worst_err}
