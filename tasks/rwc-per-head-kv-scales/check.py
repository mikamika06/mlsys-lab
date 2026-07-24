import numpy as np

from mlsys import scorers


def _decode_e4m3(code: np.ndarray) -> np.ndarray:
    """Real E4M3FN bit-pattern decode: 1 sign, 4 exponent, 3 mantissa bits."""
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return val


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_e4m3(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])  # 448.0


def _encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even encode onto the real E4M3FN grid."""
    x = np.asarray(x, dtype=np.float64)
    sign_bit = np.where(np.signbit(x), np.uint8(0x80), np.uint8(0x00))
    av = np.clip(np.abs(x), 0.0, _MAX_E4M3)
    idx = np.clip(np.searchsorted(_NONNEG_GRID, av), 1, len(_NONNEG_GRID) - 1)
    lo_idx, hi_idx = idx - 1, idx
    lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
    d_lo, d_hi = av - lo, hi - av
    hi_even = (_NONNEG_CODES[hi_idx] & 1) == 0
    choose_hi = np.where(d_hi == d_lo, hi_even, d_hi < d_lo)
    mag_code = np.where(choose_hi, _NONNEG_CODES[hi_idx], _NONNEG_CODES[lo_idx]).astype(np.uint8)
    return (sign_bit | mag_code).astype(np.uint8)


def _quant_dequant_per_head(x: np.ndarray) -> np.ndarray:
    """x: (S, H, D) -> per-head amax-scaled E4M3FN round-trip, same shape."""
    S, H, D = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for h in range(H):
        amax = np.abs(x[:, h, :]).max()
        scale = amax / _MAX_E4M3 if amax > 0 else 1.0
        codes = _encode_e4m3(x[:, h, :] / scale)
        out[:, h, :] = _decode_e4m3(codes) * scale
    return out


def _oracle(K, V, q):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    S, H, D = K.shape

    K_deq = _quant_dequant_per_head(K)
    V_deq = _quant_dequant_per_head(V)
    scale = 1.0 / np.sqrt(D)

    out = np.zeros((H, D), dtype=np.float64)
    for h in range(H):
        Kh = K_deq[:, h, :]
        Vh = V_deq[:, h, :]
        s = (Kh @ q[h]) * scale
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()
        out[h] = w @ Vh
    return out


def _build_case(rng, S, H, D, head_scales):
    """head_scales: per-head magnitude multiplier, to force divergent ranges."""
    K = rng.normal(size=(S, H, D))
    V = rng.normal(size=(S, H, D))
    for h in range(H):
        K[:, h, :] *= head_scales[h]
        V[:, h, :] *= head_scales[h] * 0.5
    q = rng.normal(size=(H, D))
    return K, V, q


def _cases():
    rng = np.random.default_rng(1337)
    specs = [
        (6, 2, 4, [1.0, 100.0]),
        (9, 3, 5, [0.5, 5.0, 50.0]),
        (4, 4, 3, [1.0, 1.0, 1.0, 1.0]),
        (12, 2, 8, [200.0, 0.1]),
    ]
    out = []
    for S, H, D, head_scales in specs:
        out.append(_build_case(rng, S, H, D, np.asarray(head_scales)))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for K, V, q in _cases():
        expected = _oracle(K, V, q)
        try:
            got = np.asarray(
                sol.per_head_kv_attention(K.copy(), V.copy(), q.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(scorers.max_abs_err(expected, got)))

    return {"max_abs_err": worst}
