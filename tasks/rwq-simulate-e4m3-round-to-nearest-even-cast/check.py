import numpy as np


def _e4m3_value_grid():
    """Return sorted array of all finite E4M3 representable values."""
    vals = set()
    for bits in range(256):
        sign = -1.0 if (bits >> 7) else 1.0
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        # NaN pattern: exp=15, mant=7
        if exp == 15 and mant == 7:
            continue  # skip NaN
        if exp == 0:
            # subnormal
            val = sign * (2 ** -6) * (mant / 8.0)
        else:
            # normal
            val = sign * (2 ** (exp - 7)) * (1.0 + mant / 8.0)
        vals.add(val)
    return np.array(sorted(vals), dtype=np.float64)


def _ref_cast_e4m3(x):
    grid = _e4m3_value_grid()
    x64 = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x64)
    flat = x64.ravel()
    res = np.empty(flat.shape, dtype=np.float64)
    for i, v in enumerate(flat):
        if np.isnan(v):
            res[i] = np.nan
            continue
        # clamp to +-448
        v_clamped = np.clip(v, -448.0, 448.0)
        diffs = np.abs(grid - v_clamped)
        min_d = diffs.min()
        candidates = np.where(diffs == min_d)[0]
        if len(candidates) == 1:
            res[i] = grid[candidates[0]]
        else:
            # tie-break: pick the one with even mantissa (or smaller magnitude as fallback)
            # For round-to-nearest-even in a sorted grid: pick even-mantissa candidate
            # Reconstruct bits for candidates
            chosen = None
            for idx in candidates:
                val = grid[idx]
                # find bits
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
                    if abs(vb - val) < 1e-15:
                        if mant_b % 2 == 0:
                            chosen = val
                            break
                if chosen is not None:
                    break
            if chosen is None:
                chosen = grid[candidates[0]]
            res[i] = chosen
    return res.reshape(x64.shape).astype(np.float32)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    # Build test inputs spanning the full range
    test_vals = []
    # subnormals
    test_vals.extend([0.0, -0.0, 2**-9, -2**-9, 2**-7, 3*2**-10])
    # normal range
    test_vals.extend([1.0, -1.0, 2.0, 0.5, 448.0, -448.0])
    # overflow
    test_vals.extend([500.0, -600.0, 1e6])
    # random
    test_vals.extend(rng.uniform(-500, 500, 200).tolist())
    test_vals.extend(rng.uniform(-1, 1, 100).tolist())
    x = np.array(test_vals, dtype=np.float32)

    ref = _ref_cast_e4m3(x)
    try:
        got = np.asarray(sol.cast_to_e4m3(x.copy()), dtype=np.float32)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != ref.shape:
        return {"exact_match": 0.0}

    # Compare: NaN vs NaN should be equal, others must be exact
    nan_mask = np.isnan(ref)
    exact = np.all(ref[~nan_mask] == got[~nan_mask]) and np.all(np.isnan(got[nan_mask]))
    return {"exact_match": 1.0 if exact else 0.0}
