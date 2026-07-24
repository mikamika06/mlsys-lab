import numpy as np

GROUP_SIZE = 32


def _oracle_quant(W, group_size):
    out_f, in_f = W.shape
    groups = in_f // group_size
    codes = np.zeros((out_f, in_f), dtype=np.uint8)
    scales = np.zeros((out_f, groups), dtype=np.float64)
    zeros = np.zeros((out_f, groups), dtype=np.float64)

    for i in range(out_f):
        for g in range(groups):
            grp = W[i, g * group_size:(g + 1) * group_size]
            wmin = float(np.min(grp))
            wmax = float(np.max(grp))
            scale = (wmax - wmin) / 15.0
            if scale == 0.0:
                scale = 1.0
            code = np.clip(np.round((grp - wmin) / scale), 0, 15).astype(np.uint8)
            codes[i, g * group_size:(g + 1) * group_size] = code
            scales[i, g] = scale
            zeros[i, g] = wmin

    return codes, scales, zeros


def _dequant(codes, scales, zeros, group_size):
    out_f, in_f = codes.shape
    groups = in_f // group_size
    W_hat = np.zeros((out_f, in_f), dtype=np.float64)
    for g in range(groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        W_hat[:, sl] = codes[:, sl].astype(np.float64) * scales[:, g:g + 1] + zeros[:, g:g + 1]
    return W_hat


def _oracle_output(W, X, group_size):
    codes, scales, zeros = _oracle_quant(W, group_size)
    W_hat = _dequant(codes, scales, zeros, group_size)
    return W_hat @ X


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(6)
    codes_ok = 1.0
    worst_err = 0.0

    for _ in range(4):
        out_f = int(rng.integers(2, 6))
        groups = int(rng.integers(1, 4))
        in_f = groups * GROUP_SIZE
        batch = int(rng.integers(2, 5))

        W = rng.standard_normal((out_f, in_f))
        X = rng.standard_normal((in_f, batch))

        ref_codes, _ref_scales, _ref_zeros = _oracle_quant(W, GROUP_SIZE)
        ref_output = _oracle_output(W, X, GROUP_SIZE)

        try:
            got = sol.int4_groupwise_asymmetric(W.copy(), X.copy(), GROUP_SIZE)
            got_codes, _got_scales, _got_zeros, got_output = got
            got_codes = np.asarray(got_codes)
            got_output = np.asarray(got_output, dtype=np.float64)
        except Exception:
            return {"codes_exact": 0.0, "max_abs_err": float("inf")}

        if got_codes.shape != ref_codes.shape or not np.array_equal(got_codes.astype(np.int64), ref_codes.astype(np.int64)):
            codes_ok = 0.0

        if got_output.shape != ref_output.shape:
            return {"codes_exact": codes_ok, "max_abs_err": float("inf")}

        worst_err = max(worst_err, float(np.max(np.abs(got_output - ref_output))))

    return {"codes_exact": codes_ok, "max_abs_err": worst_err}
