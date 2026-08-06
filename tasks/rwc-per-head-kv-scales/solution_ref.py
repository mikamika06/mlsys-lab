import math
import numpy as np


def _decode_e4m3(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    out = np.empty(code.shape, dtype=np.float64)
    it = np.nditer(code, flags=['multi_index'])
    while not it.finished:
        c = int(it[0])
        idx = it.multi_index
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07
        if e == 15 and m == 7:
            out[idx] = float('nan')
        elif e == 0:
            subnormal = sign * (m / 8.0) * math.exp2(-6.0)
            out[idx] = subnormal
        else:
            normal = sign * (1.0 + m / 8.0) * math.exp2(float(e - 7))
            out[idx] = normal
        it.iternext()
    return out


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_e4m3(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _encode_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.uint8)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        val = float(it[0])
        idx = it.multi_index
        sign_bit = 0x80 if val < 0.0 else 0x00
        av = abs(val)
        if av > _MAX_E4M3:
            av = _MAX_E4M3

        best_idx = 1
        best_diff = abs(av - float(_NONNEG_GRID[1]))
        for i in range(1, len(_NONNEG_GRID)):
            diff = abs(av - float(_NONNEG_GRID[i]))
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx == 0:
            lo_idx, hi_idx = 0, 1
        elif best_idx >= len(_NONNEG_GRID) - 1:
            lo_idx, hi_idx = len(_NONNEG_GRID) - 2, len(_NONNEG_GRID) - 1
        else:
            if float(_NONNEG_GRID[best_idx]) <= av:
                lo_idx, hi_idx = best_idx, best_idx + 1
            else:
                lo_idx, hi_idx = best_idx - 1, best_idx

        lo = float(_NONNEG_GRID[lo_idx])
        hi = float(_NONNEG_GRID[hi_idx])
        d_lo = av - lo
        d_hi = hi - av

        hi_even = (int(_NONNEG_CODES[hi_idx]) & 1) == 0
        if d_hi == d_lo:
            choose_hi = hi_even
        else:
            choose_hi = d_hi < d_lo

        mag_code = int(_NONNEG_CODES[hi_idx]) if choose_hi else int(_NONNEG_CODES[lo_idx])
        out[idx] = np.uint8(sign_bit | mag_code)
        it.iternext()
    return out


def _quant_dequant_per_head(x: np.ndarray) -> np.ndarray:
    S, H, D = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for h in range(H):
        sub = x[:, h, :]
        amax = 0.0
        it = np.nditer(sub)
        while not it.finished:
            val = abs(float(it[0]))
            if val > amax:
                amax = val
            it.iternext()
        scale = amax / _MAX_E4M3 if amax > 0 else 1.0
        codes = _encode_e4m3(sub / scale)
        out[:, h, :] = _decode_e4m3(codes) * scale
    return out


def per_head_kv_attention(K, V, q):
    """Per-head amax E4M3FN quantize/dequantize K and V, then attend.

    K, V: float64 arrays (S, H, D). q: float64 array (H, D).
    Returns float64 array (H, D).
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    S, H, D = K.shape

    K_deq = _quant_dequant_per_head(K)
    V_deq = _quant_dequant_per_head(V)
    scale = 1.0 / math.sqrt(float(D))

    out = np.zeros((H, D), dtype=np.float64)
    for h in range(H):
        Kh = K_deq[:, h, :]
        Vh = V_deq[:, h, :]
        qh = q[h]

        s = np.empty(S, dtype=np.float64)
        for s_idx in range(S):
            dot_val = 0.0
            for d_idx in range(D):
                dot_val += Kh[s_idx, d_idx] * qh[d_idx]
            s[s_idx] = dot_val * scale

        max_s = s[0]
        for s_idx in range(1, S):
            if s[s_idx] > max_s:
                max_s = s[s_idx]

        w = np.empty(S, dtype=np.float64)
        sum_w = 0.0
        for s_idx in range(S):
            val = math.exp(s[s_idx] - max_s)
            w[s_idx] = val
            sum_w += val

        for s_idx in range(S):
            w[s_idx] /= sum_w

        for d_idx in range(D):
            acc = 0.0
            for s_idx in range(S):
                acc += w[s_idx] * Vh[s_idx, d_idx]
            out[h, d_idx] = acc

    return out
