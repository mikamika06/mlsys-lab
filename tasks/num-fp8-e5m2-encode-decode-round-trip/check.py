import numpy as np

from mlsys import scorers


def _ref_decode_e5m2(codes: np.ndarray) -> np.ndarray:
    """Unambiguous closed-form E5M2 decoder: this IS the oracle."""
    codes = np.asarray(codes, dtype=np.uint8)
    s = (codes >> 7) & 1
    e = ((codes >> 2) & 0x1F).astype(np.int64)
    m = (codes & 0x3).astype(np.int64)
    sign = np.where(s == 1, -1.0, 1.0)
    normal_val = sign * np.exp2((e - 15).astype(np.float64)) * (1.0 + m / 4.0)
    sub_val = sign * np.exp2(-14.0) * (m / 4.0)
    out = np.where(e == 0, sub_val, normal_val)
    is_inf = (e == 31) & (m == 0)
    is_nan = (e == 31) & (m != 0)
    out = np.where(is_inf, sign * np.inf, out)
    out = np.where(is_nan, np.nan, out)
    return out.astype(np.float32)


# Ascending magnitudes for codes 0..123 (all finite, nonnegative, e in [0,30]).
_MAG_GRID = _ref_decode_e5m2(np.arange(124, dtype=np.uint8)).astype(np.float64)
_MAX_FINITE = float(_MAG_GRID[-1])          # 57344.0
_NEXT_VIRTUAL = 65536.0                     # 2^16: where the exponent field would
                                             # continue if e=31 were not reserved.
_EXT_GRID = np.concatenate([_MAG_GRID, [_NEXT_VIRTUAL]])


def _ref_encode_e5m2(values: np.ndarray) -> np.ndarray:
    """Reference E5M2 encoder derived purely from _ref_decode_e5m2's grid:
    nearest-grid-point rounding with ties-to-even, extended one virtual step
    past the max finite value so the infinity-overflow boundary follows the
    same round-to-nearest rule as every other step.
    """
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    shape = values.shape
    flat = values.ravel()
    sign_bit = np.signbit(flat).astype(np.uint8)
    av = np.abs(flat)

    out = np.zeros(flat.shape, dtype=np.uint8)

    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    finite_mask = ~(nan_mask | inf_mask)

    if np.any(finite_mask):
        av_f = av[finite_mask]
        av_c = np.minimum(av_f, _NEXT_VIRTUAL)
        idx = np.searchsorted(_EXT_GRID, av_c, side="left")
        idx = np.clip(idx, 1, len(_EXT_GRID) - 1)
        lo = idx - 1
        hi = idx
        d_lo = av_c - _EXT_GRID[lo]
        d_hi = _EXT_GRID[hi] - av_c
        pick_hi = d_hi < d_lo
        tie = d_hi == d_lo
        hi_even = (hi % 2) == 0
        pick_hi = np.where(tie, hi_even, pick_hi)
        chosen = np.where(pick_hi, hi, lo)
        chosen = np.where(av_f > _NEXT_VIRTUAL, len(_EXT_GRID) - 1, chosen)
        is_inf_result = chosen >= (len(_EXT_GRID) - 1)
        code_finite = np.where(is_inf_result, 0x7C, chosen).astype(np.uint8)
        out[finite_mask] = code_finite

    out[inf_mask] = 0x7C
    out[nan_mask] = 0x7F

    out = (sign_bit << 7) | out
    return out.reshape(shape)


def _build_test_values() -> np.ndarray:
    rng = np.random.default_rng(0)

    # Every exactly-representable finite grid value (both signs) -> must
    # round-trip through encode_e5m2 to its own code.
    grid_vals = _MAG_GRID
    exact = np.concatenate([grid_vals, -grid_vals[1:]])  # skip duplicate -0.0 of index0

    # RNE tie points: exact midpoints between adjacent grid codes.
    ties = (_MAG_GRID[:-1] + _MAG_GRID[1:]) / 2.0
    ties = np.concatenate([ties, -ties])

    # Overflow boundary region (max finite=57344, virtual=65536, midpoint=61440).
    boundary = np.array([57344.0, 57345.0, 60000.0, 61439.0, 61440.0, 61441.0,
                          65535.0, 65536.0, 70000.0, 1e9])
    boundary = np.concatenate([boundary, -boundary])

    # Random continuous values across many scales, including subnormal range.
    rand = np.concatenate([
        rng.normal(0.0, 3.0, size=4000),
        rng.uniform(-6e-5, 6e-5, size=2000),      # near subnormal range
        rng.uniform(-2000.0, 2000.0, size=4000),
        rng.uniform(-70000.0, 70000.0, size=3000),
    ])

    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan])

    x = np.concatenate([exact, ties, boundary, rand, specials]).astype(np.float32)
    return x


def grade(sol, fx) -> dict:
    x = _build_test_values()
    ref_codes = _ref_encode_e5m2(x)

    x_in = x.copy()
    try:
        got = sol.encode_e5m2(x_in)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    if not np.array_equal(x_in, x, equal_nan=True):
        return {"byte_exact_fraction": 0.0}

    got = np.asarray(got)
    if got.shape != ref_codes.shape:
        return {"byte_exact_fraction": 0.0}
    try:
        got = got.astype(np.uint8)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    frac = scorers.byte_exact_fraction(ref_codes, got)
    return {"byte_exact_fraction": frac}
