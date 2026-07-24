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


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even cast to the real E4M3FN grid, saturating at
    +-448."""
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


def _quant_dequant(x: np.ndarray, scale: float) -> np.ndarray:
    return _e4m3_round_trip(x / scale) * scale


def _attention(K: np.ndarray, V: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = K.shape[1]
    logits = (K @ q) / np.sqrt(d)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return w @ V


def per_head_scale_attention_errors(
    K: np.ndarray, V: np.ndarray, q: np.ndarray, percentile: float
) -> np.ndarray:
    """
    K, V: (n, d) fp64 -- one attention head's key/value cache.
    q: (d,) fp64 query vector.
    percentile: percentile (0-100) used for the percentile-clipped scale.

    Quantize K to FP8 E4M3 twice with two different per-head scales:
      - amax scale:       scale = max(|K|) / 448
      - percentile scale:  scale = percentile(|K|, percentile) / 448
                            (values beyond this saturate to +-448 rather
                            than being represented exactly)

    For each variant, dequantize K and compute the exact softmax
    attention output softmax(K_hat @ q / sqrt(d)) @ V using the exact
    fp64 V, then compare against the exact fp64 attention output computed
    from the unquantized K.

    Returns np.array([attn_max_abs_err_amax, attn_max_abs_err_percentile]).
    """
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
