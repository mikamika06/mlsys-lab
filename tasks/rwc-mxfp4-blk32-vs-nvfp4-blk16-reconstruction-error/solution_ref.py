import numpy as np

_FP4_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MAX = 6.0


def _snap_fp4(mag: np.ndarray) -> np.ndarray:
    idx = np.abs(mag[..., None] - _FP4_LEVELS[None, :]).argmin(axis=-1)
    return _FP4_LEVELS[idx]


def _mxfp4_quant_dequant(x: np.ndarray, block: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    n = len(flat)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, block):
        blk = flat[i:i + block]
        amax = float(np.max(np.abs(blk)))
        if amax == 0.0:
            scale = 1.0
        else:
            scale = 2.0 ** np.ceil(np.log2(amax / _FP4_MAX))
        sign = np.sign(blk)
        mag = np.clip(np.abs(blk) / scale, 0.0, _FP4_MAX)
        out[i:i + block] = sign * _snap_fp4(mag) * scale
    return out.reshape(x.shape)


def _e4m3_decode(code):
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
_NONNEG_GRID = _e4m3_decode(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(np.signbit(x), -1.0, 1.0)
    av = np.clip(np.abs(x), 0.0, _MAX_E4M3)
    idx = np.searchsorted(_NONNEG_GRID, av)
    idx = np.clip(idx, 1, len(_NONNEG_GRID) - 1)
    lo_idx, hi_idx = idx - 1, idx
    lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
    d_lo, d_hi = av - lo, hi - av
    hi_even = (_NONNEG_CODES[hi_idx] & 1) == 0
    choose_hi = np.where(d_hi == d_lo, hi_even, d_hi < d_lo)
    chosen = np.where(choose_hi, hi, lo)
    result = sign * chosen
    result = np.where(x == 0, np.copysign(0.0, x), result)
    return result


def _nvfp4_quant_dequant(x: np.ndarray, block: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    n = len(flat)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, block):
        blk = flat[i:i + block]
        amax = float(np.max(np.abs(blk)))
        if amax == 0.0:
            scale = 1.0
        else:
            scale_real = amax / _FP4_MAX
            scale = float(_e4m3_round_trip(np.array([scale_real]))[0])
            if scale == 0.0:
                scale = scale_real
        sign = np.sign(blk)
        mag = np.clip(np.abs(blk) / scale, 0.0, _FP4_MAX)
        out[i:i + block] = sign * _snap_fp4(mag) * scale
    return out.reshape(x.shape)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def compare_mxfp4_nvfp4(weights: np.ndarray) -> np.ndarray:
    """
    weights: array of any shape.

    Quantize-then-dequantize `weights` with both 4-bit microscaling
    schemes (both use the same E2M1 FP4 grid [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    per element, only the shared block scale differs):

    - MXFP4: block of 32 elements, shared scale restricted to a power of
      two (an E8M0-style exponent-only scale) -- the largest power of
      two that keeps the block's amax within the FP4 grid's max
      magnitude (6.0).
    - NVFP4: block of 16 elements, shared scale computed as
      amax/6.0 and then rounded to the nearest real FP8 E4M3 value (not
      restricted to a power of two).

    Returns np.array([mxfp4_rel_err, nvfp4_rel_err]), the global relative
    L2 reconstruction error ||dequantized - weights|| / ||weights|| for
    each scheme against the original weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    mx = _mxfp4_quant_dequant(weights, block=32)
    nv = _nvfp4_quant_dequant(weights, block=16)
    return np.array([_rel_err(weights, mx), _rel_err(weights, nv)])
