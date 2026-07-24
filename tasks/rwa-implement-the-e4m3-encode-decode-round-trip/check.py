import numpy as np


def _decode_bits(code: np.ndarray) -> np.ndarray:
    """Real E4M3FN bit-pattern decode: 1 sign, 4 exponent, 3 mantissa bits."""
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)  # reserved NaN code
    return val


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)  # excludes 0x7F (NaN)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)          # ascending by construction
_MAX_E4M3 = float(_NONNEG_GRID[-1])                 # 448.0


def _oracle_round_trip(x: np.ndarray) -> np.ndarray:
    """Reference encode-then-decode: clamp to +-448, round-to-nearest-even
    against the real 128-point nonnegative E4M3FN grid, restore sign.
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
    result = np.where(x == 0, np.copysign(0.0, x), result)  # preserve +-0
    return result.astype(np.float32)


def _fixed_probe_array() -> np.ndarray:
    """A fixed (non-random), diverse set of probe values: zero (+/-),
    values exactly on the grid, RNE tie-break midpoints in both the
    subnormal and normal ranges, ordinary off-grid values needing simple
    rounding, values right at/just past the +-448 saturation boundary,
    and negative counterparts.
    """
    on_grid = _NONNEG_GRID[[0, 1, 4, 20, 60, 90, 126]]

    # RNE tie midpoints: between codes 3<->4 (subnormal) and 40<->41, 80<->81
    # (normal), one of each parity pairing so both round-to-even directions
    # get exercised.
    tie_pairs = [(3, 4), (4, 5), (40, 41), (41, 42), (80, 81), (81, 82)]
    ties = np.array(
        [(_NONNEG_GRID[a] + _NONNEG_GRID[b]) / 2.0 for a, b in tie_pairs]
    )

    # Ordinary off-grid values (simple nearest-neighbor rounding, no tie).
    off_grid = _NONNEG_GRID[[10, 50, 100]] * 1.1

    saturating = np.array([448.0, 448.5, 500.0, 10000.0])

    positive = np.concatenate([on_grid, ties, off_grid, saturating])
    values = np.concatenate([[0.0, -0.0], positive, -positive])
    return values.astype(np.float64)


def grade(sol, fx) -> dict:
    x = _fixed_probe_array()
    ref = _oracle_round_trip(x)

    try:
        got = sol.e4m3_round_trip(x.copy())
    except Exception:
        return {"max_abs_err": float("inf")}

    got = np.asarray(got)
    if got.shape != ref.shape:
        return {"max_abs_err": float("inf")}

    try:
        got64 = got.astype(np.float64)
    except Exception:
        return {"max_abs_err": float("inf")}

    err = float(np.max(np.abs(got64 - ref.astype(np.float64))))
    return {"max_abs_err": err}
