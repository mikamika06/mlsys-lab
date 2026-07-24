import numpy as np


def _make_2_4_case(rng, d_out, groups):
    d_in = groups * 4
    half = d_in // 2
    W = rng.normal(size=(d_out, d_in))

    mask = np.zeros_like(W, dtype=bool)
    for g in range(groups):
        block = np.abs(W[:, g * 4:(g + 1) * 4])
        order = np.argsort(-block, axis=1, kind="stable")
        keep = order[:, :2]
        rows = np.arange(d_out)[:, None]
        mask[rows, g * 4 + keep] = True
    W_masked = W * mask

    values = np.zeros((d_out, half))
    idx = np.zeros((d_out, half), dtype=np.int64)
    for g in range(groups):
        block_mask = mask[:, g * 4:(g + 1) * 4]  # (d_out, 4)
        for r in range(d_out):
            positions = np.nonzero(block_mask[r])[0]  # ascending, len 2
            values[r, g * 2:g * 2 + 2] = W_masked[r, g * 4 + positions]
            idx[r, g * 2:g * 2 + 2] = positions

    return W_masked, values, idx


def _oracle_reconstruct(values, idx):
    d_out, half = values.shape
    d_in = half * 2
    row_idx = np.broadcast_to(np.arange(d_out)[:, None], (d_out, half))
    group_base = (np.arange(half) // 2) * 4
    cols = group_base[None, :] + idx
    W = np.zeros((d_out, d_in))
    W[row_idx, cols] = values
    return W


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        d_out = int(rng.integers(2, 8))
        groups = int(rng.integers(2, 6))
        n = int(rng.integers(1, 6))

        W_masked, values, idx = _make_2_4_case(rng, d_out, groups)
        d_in = groups * 4
        X = rng.normal(size=(d_in, n))

        Y_exp = W_masked @ X
        # sanity: the fixture's oracle reconstruction must round-trip exactly
        assert np.array_equal(_oracle_reconstruct(values, idx), W_masked)

        try:
            Y_got = np.asarray(
                sol.compressed_matmul(values.copy(), idx.copy(), X.copy()), dtype=np.float64
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if Y_got.shape != Y_exp.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(Y_got - Y_exp))))

    return {"max_abs_err": worst}
