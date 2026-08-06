import math
import numpy as np


def _decode_bits(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    out = np.empty(code.shape, dtype=np.float64)
    flat_code = code.ravel()
    flat_out = out.ravel()

    for i in range(flat_code.shape[0]):
        c = int(flat_code[i])
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07

        if e == 15 and m == 7:
            val = float('nan')
        elif e == 0:
            subnormal = sign * (m / 8.0) * math.exp2(-6.0)
            val = subnormal
        else:
            normal = sign * (1.0 + m / 8.0) * math.exp2(float(e - 7))
            val = normal

        flat_out[i] = val

    return out


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)
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
    out = np.empty(x.shape, dtype=np.float32)
    flat_x = x.ravel()
    flat_out = out.ravel()

    grid = _NONNEG_GRID
    codes = _NONNEG_CODES
    grid_len = grid.shape[0]

    for i in range(flat_x.shape[0]):
        val = flat_x[i]

        # signbit check
        if math.copysign(1.0, val) < 0.0 or (val == 0.0 and 1.0 / val < 0.0):
            sign = -1.0
        else:
            sign = 1.0

        # abs and clamp
        av = abs(val)
        if av > _MAX_E4M3:
            av = _MAX_E4M3

        # searchsorted equivalent
        idx = grid_len
        for j in range(grid_len):
            if grid[j] >= av:
                idx = j
                break

        # clip idx
        if idx < 1:
            idx = 1
        elif idx > grid_len - 1:
            idx = grid_len - 1

        lo_idx = idx - 1
        hi_idx = idx
        lo = grid[lo_idx]
        hi = grid[hi_idx]

        d_lo = av - lo
        d_hi = hi - av

        hi_code_even = (int(codes[hi_idx]) & 1) == 0

        if d_hi == d_lo:
            choose_hi = hi_code_even
        else:
            choose_hi = d_hi < d_lo

        chosen = hi if choose_hi else lo
        result = sign * chosen

        if val == 0.0:
            if sign < 0.0:
                result = -0.0
            else:
                result = 0.0

        flat_out[i] = np.float32(result)

    return out
