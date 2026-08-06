import math
import numpy as np

_FP4_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MAX = 6.0


def _snap_fp4(mag: np.ndarray) -> np.ndarray:
    mag_flat = mag.ravel()
    res = np.empty_like(mag_flat)
    for i in range(len(mag_flat)):
        val = mag_flat[i]
        best_level = _FP4_LEVELS[0]
        best_dist = abs(val - best_level)
        for level in _FP4_LEVELS[1:]:
            dist = abs(val - level)
            if dist < best_dist:
                best_dist = dist
                best_level = level
        res[i] = best_level
    return res.reshape(mag.shape)


def _mxfp4_quant_dequant(x: np.ndarray, block: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    n = len(flat)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, block):
        blk = flat[i:i + block]
        amax = 0.0
        for val in blk:
            aval = abs(val)
            if aval > amax:
                amax = aval
        
        if amax == 0.0:
            scale = 1.0
        else:
            scale = 2.0 ** math.ceil(math.log2(amax / _FP4_MAX))
            
        for j in range(len(blk)):
            v = blk[j]
            sign = -1.0 if v < 0.0 else (1.0 if v > 0.0 else 0.0)
            m = abs(v) / scale
            if m < 0.0:
                m = 0.0
            elif m > _FP4_MAX:
                m = _FP4_MAX
                
            best_level = _FP4_LEVELS[0]
            best_dist = abs(m - best_level)
            for level in _FP4_LEVELS[1:]:
                dist = abs(m - level)
                if dist < best_dist:
                    best_dist = dist
                    best_level = level
            out[i + j] = sign * best_level * scale
    return out.reshape(x.shape)


def _e4m3_decode(code):
    code = np.asarray(code, dtype=np.uint8)
    flat_code = code.ravel()
    res = np.empty(len(flat_code), dtype=np.float64)
    for i in range(len(flat_code)):
        c = int(flat_code[i])
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07
        if e == 15 and m == 7:
            res[i] = float('nan')
        elif e == 0:
            res[i] = sign * (m / 8.0) * math.exp2(-6.0)
        else:
            res[i] = sign * (1.0 + m / 8.0) * math.exp2(e - 7)
    return res.reshape(code.shape)


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _e4m3_decode(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    res = np.empty(len(flat), dtype=np.float64)
    for k in range(len(flat)):
        val = flat[k]
        sign = -1.0 if math.copysign(1.0, val) < 0 else 1.0
        av = abs(val)
        if av < 0.0:
            av = 0.0
        elif av > _MAX_E4M3:
            av = _MAX_E4M3
            
        idx = 0
        while idx < len(_NONNEG_GRID) and _NONNEG_GRID[idx] < av:
            idx += 1
        if idx >= len(_NONNEG_GRID):
            idx = len(_NONNEG_GRID) - 1
        if idx < 1:
            idx = 1
            
        lo_idx, hi_idx = idx - 1, idx
        lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
        d_lo, d_hi = av - lo, hi - av
        hi_even = (int(_NONNEG_CODES[hi_idx]) & 1) == 0
        
        if d_hi == d_lo:
            choose_hi = hi_even
        else:
            choose_hi = d_hi < d_lo
            
        chosen = hi if choose_hi else lo
        r = sign * chosen
        if val == 0:
            r = math.copysign(0.0, val)
        res[k] = r
    return res.reshape(x.shape)


def _nvfp4_quant_dequant(x: np.ndarray, block: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    n = len(flat)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, block):
        blk = flat[i:i + block]
        amax = 0.0
        for val in blk:
            aval = abs(val)
            if aval > amax:
                amax = aval
                
        if amax == 0.0:
            scale = 1.0
        else:
            scale_real = amax / _FP4_MAX
            scale = float(_e4m3_round_trip(np.array([scale_real]))[0])
            if scale == 0.0:
                scale = scale_real
                
        for j in range(len(blk)):
            v = blk[j]
            sign = -1.0 if v < 0.0 else (1.0 if v > 0.0 else 0.0)
            m = abs(v) / scale
            if m < 0.0:
                m = 0.0
            elif m > _FP4_MAX:
                m = _FP4_MAX
                
            best_level = _FP4_LEVELS[0]
            best_dist = abs(m - best_level)
            for level in _FP4_LEVELS[1:]:
                dist = abs(m - level)
                if dist < best_dist:
                    best_dist = dist
                    best_level = level
            out[i + j] = sign * best_level * scale
    return out.reshape(x.shape)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    
    sum_diff_sq = 0.0
    sum_a_sq = 0.0
    for i in range(len(a)):
        diff = b[i] - a[i]
        sum_diff_sq += diff * diff
        sum_a_sq += a[i] * a[i]
        
    norm_diff = math.sqrt(sum_diff_sq)
    norm_a = math.sqrt(sum_a_sq)
    return float(norm_diff / (norm_a + 1e-12))


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
