import numpy as np


def _decode_bits(code: np.ndarray) -> np.ndarray:
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


def e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    """
    Simulate encoding `x` to E4M3FN and decoding it back:
    - clamp magnitude to +-448 (saturation),
    - round-to-nearest-even against the real E4M3FN grid (subnormals
      included via the exponent-field-0 branch of the format),
    - preserve the sign of zero.
    """
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
    return result.astype(np.float32)
