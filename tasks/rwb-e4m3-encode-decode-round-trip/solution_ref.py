import math
import numpy as np


def _decode_bits(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    flat_code = code.ravel()
    out = np.empty(flat_code.shape, dtype=np.float64)
    for i in range(flat_code.shape[0]):
        c = int(flat_code[i])
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07
        if e == 15 and m == 7:
            out[i] = float("nan")
        elif e == 0:
            subnormal = sign * (m / 8.0) * math.exp2(-6.0)
            out[i] = subnormal
        else:
            normal = sign * (1.0 + m / 8.0) * math.exp2(float(e - 7))
            out[i] = normal
    return out.reshape(code.shape)


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    """Decode raw E4M3FN byte patterns to float values (NaN for the
    reserved S.1111.111 code, either sign)."""
    return _decode_bits(codes).astype(np.float32)


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encode floats to E4M3FN byte patterns: saturate to +-448, then
    round-to-nearest-even against the real 128-point nonnegative grid."""
    x = np.asarray(x, dtype=np.float64)
    flat_x = x.ravel()
    out = np.empty(flat_x.shape, dtype=np.uint8)
    
    grid = _NONNEG_GRID
    codes = _NONNEG_CODES
    max_grid_idx = len(grid) - 1

    for i in range(flat_x.shape[0]):
        val = flat_x[i]
        sign_bit = 0x80 if math.copysign(1.0, val) < 0 else 0x00
        av = abs(val)
        if av > _MAX_E4M3:
            av = _MAX_E4M3

        low = 0
        high = len(grid)
        while low < high:
            mid = (low + high) // 2
            if grid[mid] < av:
                low = mid + 1
            else:
                high = mid
        idx = low
        if idx < 1:
            idx = 1
        elif idx > max_grid_idx:
            idx = max_grid_idx

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

        chosen_code = codes[hi_idx] if choose_hi else codes[lo_idx]
        out[i] = sign_bit | int(chosen_code)

    return out.reshape(x.shape)
