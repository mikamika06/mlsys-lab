import numpy as np


def _oracle(W, group_size):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    n_groups = cols // group_size
    Wg = W.reshape(rows, n_groups, group_size)

    amax = np.max(np.abs(Wg), axis=-1)  # (rows, n_groups)
    scales = np.where(amax == 0, 1.0, amax / 8.0)

    codes_g = np.clip(np.round(Wg / scales[:, :, None]), -8, 7).astype(np.int64)
    codes = codes_g.reshape(rows, cols)
    return codes, scales


def _dequant(codes, scales, group_size):
    codes = np.asarray(codes, dtype=np.float64)
    rows, cols = codes.shape
    n_groups = cols // group_size
    codes_g = codes.reshape(rows, n_groups, group_size)
    return (codes_g * np.asarray(scales, dtype=np.float64)[:, :, None]).reshape(rows, cols)


def _case(seed, rows, n_groups, group_size, scale_spread=1.0):
    rng = np.random.default_rng(seed)
    cols = n_groups * group_size
    # vary per-group magnitude so different groups get very different
    # scales, exercising genuinely independent per-group quantization.
    group_mag = np.exp(rng.uniform(-scale_spread, scale_spread, size=(rows, n_groups)))
    W = rng.standard_normal((rows, n_groups, group_size)) * group_mag[:, :, None]
    W = W.reshape(rows, cols)
    return W, group_size


def grade(sol, fx) -> dict:
    cases = [
        _case(1, 4, 3, 32),
        _case(2, 8, 2, 32, scale_spread=2.0),
        _case(3, 2, 5, 32, scale_spread=0.5),
        _case(4, 3, 1, 32),
    ]
    # explicit all-zero group -> scale must fall back to 1.0, not divide by 0.
    W_zero = np.zeros((2, 64))
    cases.append((W_zero, 32))

    exact = 1.0
    for W, group_size in cases:
        ref_codes, ref_scales = _oracle(W, group_size)
        try:
            got_codes, got_scales = sol.int4_groupwise_quant(W.copy(), group_size)
            got_codes = np.asarray(got_codes, dtype=np.int64)
            got_scales = np.asarray(got_scales, dtype=np.float64)
        except Exception:
            exact = 0.0
            break

        if got_codes.shape != ref_codes.shape or got_scales.shape != ref_scales.shape:
            exact = 0.0
            break
        if not np.array_equal(got_codes, ref_codes):
            exact = 0.0
            break
        if not np.allclose(got_scales, ref_scales, rtol=1e-9, atol=1e-12):
            exact = 0.0
            break
        if np.any(got_codes < -8) or np.any(got_codes > 7):
            exact = 0.0
            break
        # dequantized reconstruction must also match within a tight tolerance
        got_deq = _dequant(got_codes, got_scales, group_size)
        ref_deq = _dequant(ref_codes, ref_scales, group_size)
        if not np.allclose(got_deq, ref_deq, rtol=1e-9, atol=1e-9):
            exact = 0.0
            break

    return {"exact_match": exact}
