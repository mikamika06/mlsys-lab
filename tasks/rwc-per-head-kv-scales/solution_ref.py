import numpy as np


def _decode_e4m3(code: np.ndarray) -> np.ndarray:
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
    S, H, D = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for h in range(H):
        amax = np.abs(x[:, h, :]).max()
        scale = amax / _MAX_E4M3 if amax > 0 else 1.0
        codes = _encode_e4m3(x[:, h, :] / scale)
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
