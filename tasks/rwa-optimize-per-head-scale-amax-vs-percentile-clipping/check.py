import numpy as np


def _decode_bits(code):
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return val


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)  # excludes 0x7F (NaN)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)          # ascending, grid[-1] == 448
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _e4m3_round_trip(x):
    """Round-to-nearest-even cast to the real E4M3FN grid, with saturation
    at +-448 (this is a full, exact FP8 E4M3 simulator, not an
    approximation)."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(np.signbit(x), -1.0, 1.0)
    av = np.clip(np.abs(x), 0.0, _MAX_E4M3)
    idx = np.searchsorted(_NONNEG_GRID, av)
    idx = np.clip(idx, 1, len(_NONNEG_GRID) - 1)
    lo_idx, hi_idx = idx - 1, idx
    lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
    d_lo, d_hi = av - lo, hi - av
    hi_code_even = (_NONNEG_CODES[hi_idx] & 1) == 0
    choose_hi = np.where(d_hi == d_lo, hi_code_even, d_hi < d_lo)
    chosen = np.where(choose_hi, hi, lo)
    result = sign * chosen
    result = np.where(x == 0, np.copysign(0.0, x), result)
    return result


def _quant_dequant(x, scale):
    return _e4m3_round_trip(x / scale) * scale


def _attention(K, V, q):
    d = K.shape[1]
    logits = (K @ q) / np.sqrt(d)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return w @ V


def _oracle(K, V, q, percentile):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    amax = float(np.max(np.abs(K)))
    p = float(np.percentile(np.abs(K), percentile))

    scale_amax = amax / 448.0
    scale_pct = p / 448.0

    K_amax = _quant_dequant(K, scale_amax)
    K_pct = _quant_dequant(K, scale_pct)

    base = _attention(K, V, q)
    out_amax = _attention(K_amax, V, q)
    out_pct = _attention(K_pct, V, q)

    err_amax = float(np.max(np.abs(out_amax - base)))
    err_pct = float(np.max(np.abs(out_pct - base)))
    return np.array([err_amax, err_pct])


def _make_case(seed, n, d, outlier_mag):
    """One key (row 0) is set to a huge, deeply-off-query outlier: it
    points strongly AWAY from q (very negative logit), so its own softmax
    weight is negligible in the exact computation and stays negligible
    under quantization noise too -- but its raw magnitude dominates
    amax, forcing every "normal" key onto a much coarser amax-derived
    scale than a percentile-derived scale would use."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))
    q = rng.standard_normal(d)
    q_hat = q / np.linalg.norm(q)
    K[0] = -outlier_mag * q_hat
    return K, V, q


def _cases():
    # (seed, n, d, outlier_mag, percentile) -- each hand-picked so the
    # oracle's percentile-clipped scale gives a clearly lower attention
    # output error than the amax scale.
    specs = [
        (195, 400, 8, 10000.0, 99.5),
        (11, 1000, 10, 20000.0, 99.9),
        (166, 400, 8, 10000.0, 99.5),
        (21, 300, 16, 5000.0, 99.5),
    ]
    cases = []
    for seed, n, d, mag, pct in specs:
        K, V, q = _make_case(seed, n, d, mag)
        cases.append((K, V, q, pct))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for K, V, q, pct in _cases():
        ref = _oracle(K, V, q, pct)
        assert ref[1] < ref[0]  # percentile clipping must win on these cases

        try:
            got = np.asarray(
                sol.per_head_scale_attention_errors(K.copy(), V.copy(), q.copy(), pct),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
