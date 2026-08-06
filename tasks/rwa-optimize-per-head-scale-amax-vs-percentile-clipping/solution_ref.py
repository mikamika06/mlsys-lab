import math
import numpy as np


def _decode_bits(code):
    code = np.asarray(code, dtype=np.uint8)
    res = np.empty(code.shape, dtype=np.float64)
    flat_code = code.ravel()
    flat_res = res.ravel()
    for i in range(flat_code.size):
        c = int(flat_code[i])
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07
        if e == 15 and m == 7:
            flat_res[i] = float('nan')
        elif e == 0:
            flat_res[i] = sign * (m / 8.0) * math.exp2(-6.0)
        else:
            flat_res[i] = sign * (1.0 + m / 8.0) * math.exp2(float(e - 7))
    return res


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even cast to the real E4M3FN grid, saturating at
    +-448."""
    x = np.asarray(x, dtype=np.float64)
    res = np.empty(x.shape, dtype=np.float64)
    flat_x = x.ravel()
    flat_res = res.ravel()
    
    for i in range(flat_x.size):
        val = flat_x[i]
        if math.isnan(val):
            flat_res[i] = float('nan')
            continue
            
        sign = -1.0 if math.copysign(1.0, val) < 0.0 else 1.0
        av = abs(val)
        if av > _MAX_E4M3:
            av = _MAX_E4M3
            
        idx = 0
        for j in range(len(_NONNEG_GRID)):
            if _NONNEG_GRID[j] >= av:
                idx = j
                break
        else:
            idx = len(_NONNEG_GRID) - 1
            
        if idx < 1:
            idx = 1
        elif idx >= len(_NONNEG_GRID):
            idx = len(_NONNEG_GRID) - 1
            
        lo_idx, hi_idx = idx - 1, idx
        lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
        d_lo, d_hi = av - lo, hi - av
        
        hi_code_even = (int(_NONNEG_CODES[hi_idx]) & 1) == 0
        if d_hi == d_lo:
            choose_hi = hi_code_even
        else:
            choose_hi = d_hi < d_lo
            
        chosen = hi if choose_hi else lo
        r = sign * chosen
        if val == 0.0:
            r = math.copysign(0.0, val)
        flat_res[i] = r
    return res


def _quant_dequant(x: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.empty(x.shape, dtype=np.float64)
    flat_x = x.ravel()
    flat_scaled = scaled.ravel()
    for i in range(flat_x.size):
        flat_scaled[i] = flat_x[i] / scale
    rounded = _e4m3_round_trip(scaled)
    out = np.empty(x.shape, dtype=np.float64)
    flat_out = out.ravel()
    flat_rounded = rounded.ravel()
    for i in range(flat_rounded.size):
        flat_out[i] = flat_rounded[i] * scale
    return out


def _attention(K: np.ndarray, V: np.ndarray, q: np.ndarray) -> np.ndarray:
    n, d = K.shape
    logits = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            s += K[i, j] * q[j]
        logits[i] = s / math.sqrt(float(d))
        
    max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]
            
    w = np.empty(n, dtype=np.float64)
    sum_w = 0.0
    for i in range(n):
        val = math.exp(logits[i] - max_logit)
        w[i] = val
        sum_w += val
        
    for i in range(n):
        w[i] /= sum_w
        
    v_cols = V.shape[1]
    res = np.zeros(v_cols, dtype=np.float64)
    for j in range(v_cols):
        s = 0.0
        for i in range(n):
            s += w[i] * V[i, j]
        res[j] = s
    return res


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

    flat_k = K.ravel()
    amax = 0.0
    for i in range(flat_k.size):
        val = abs(flat_k[i])
        if val > amax:
            amax = val
    amax = float(amax)

    abs_k = np.empty(flat_k.size, dtype=np.float64)
    for i in range(flat_k.size):
        abs_k[i] = abs(flat_k[i])
    
    sorted_abs = sorted(abs_k)
    n_elems = len(sorted_abs)
    idx_pct = (percentile / 100.0) * (n_elems - 1)
    lower_idx = int(math.floor(idx_pct))
    upper_idx = int(math.ceil(idx_pct))
    if lower_idx == upper_idx:
        p = float(sorted_abs[lower_idx])
    else:
        weight = idx_pct - lower_idx
        p = float(sorted_abs[lower_idx] * (1.0 - weight) + sorted_abs[upper_idx] * weight)

    scale_amax = amax / 448.0
    scale_pct = p / 448.0

    K_amax = _quant_dequant(K, scale_amax)
    K_pct = _quant_dequant(K, scale_pct)

    base = _attention(K, V, q)
    out_amax = _attention(K_amax, V, q)
    out_pct = _attention(K_pct, V, q)

    max_err_amax = 0.0
    flat_amax_out = out_amax.ravel()
    flat_base = base.ravel()
    for i in range(flat_base.size):
        diff = abs(flat_amax_out[i] - flat_base[i])
        if diff > max_err_amax:
            max_err_amax = diff

    max_err_pct = 0.0
    flat_pct_out = out_pct.ravel()
    for i in range(flat_base.size):
        diff = abs(flat_pct_out[i] - flat_base[i])
        if diff > max_err_pct:
            max_err_pct = diff

    return np.array([float(max_err_amax), float(max_err_pct)])
