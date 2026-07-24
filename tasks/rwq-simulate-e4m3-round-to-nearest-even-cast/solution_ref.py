import numpy as np


def _e4m3_value_grid():
    """Return array of all finite E4M3 representable values."""
    vals = set()
    for bits in range(256):
        sign = -1.0 if (bits >> 7) else 1.0
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        if exp == 15 and mant == 7:
            continue
        if exp == 0:
            val = sign * (2 ** -6) * (mant / 8.0)
        else:
            val = sign * (2 ** (exp - 7)) * (1.0 + mant / 8.0)
        vals.add(val)
    return np.array(sorted(vals), dtype=np.float64)


def cast_to_e4m3(x):
    """Cast float32 array to nearest E4M3 value (round-to-nearest-even, clamp at +-448)."""
    grid = _e4m3_value_grid()
    x64 = np.asarray(x, dtype=np.float64)
    flat = x64.ravel()
    res = np.empty(flat.shape, dtype=np.float64)
    for i, v in enumerate(flat):
        if np.isnan(v):
            res[i] = np.nan
            continue
        v_clamped = float(np.clip(v, -448.0, 448.0))
        diffs = np.abs(grid - v_clamped)
        min_d = diffs.min()
        candidates = np.where(diffs == min_d)[0]
        if len(candidates) == 1:
            res[i] = grid[candidates[0]]
        else:
            # tie-break: pick even mantissa
            chosen = None
            for idx in candidates:
                val = grid[idx]
                for bits in range(256):
                    sign_b = -1.0 if (bits >> 7) else 1.0
                    exp_b = (bits >> 3) & 0xF
                    mant_b = bits & 0x7
                    if exp_b == 15 and mant_b == 7:
                        continue
                    if exp_b == 0:
                        vb = sign_b * (2 ** -6) * (mant_b / 8.0)
                    else:
                        vb = sign_b * (2 ** (exp_b - 7)) * (1.0 + mant_b / 8.0)
                    if abs(vb - val) < 1e-15 and mant_b % 2 == 0:
                        chosen = val
                        break
                if chosen is not None:
                    break
            res[i] = chosen if chosen is not None else grid[candidates[0]]
    return res.reshape(x64.shape).astype(np.float32)
