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
    """Saturate to +-448 and round-to-nearest-even to the real grid."""
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
    return result


FAIL = {"max_abs_err": float("inf")}


def grade(sol, fx) -> dict:
    # 1) decode: every one of the 256 possible byte values.
    all_codes = np.arange(256, dtype=np.uint8)
    ref_decoded = _decode_bits(all_codes)

    try:
        got_decoded = np.asarray(sol.decode_e4m3(all_codes.copy()), dtype=np.float64)
    except Exception:
        return dict(FAIL)

    if got_decoded.shape != ref_decoded.shape:
        return dict(FAIL)

    ref_nan = np.isnan(ref_decoded)
    got_nan = np.isnan(got_decoded)
    if not np.array_equal(ref_nan, got_nan):
        return dict(FAIL)

    diff_decode = np.abs(got_decoded[~ref_nan] - ref_decoded[~ref_nan])
    err_decode = float(diff_decode.max()) if diff_decode.size else 0.0

    # 2) encode: probe values from the fixture, decoded via the *verified*
    #    bit-pattern formula above (independent of the student's own decode).
    x = np.asarray(fx["x"], dtype=np.float64)
    ref_roundtrip = _oracle_round_trip(x)

    try:
        got_codes = np.asarray(sol.encode_e4m3(x.astype(np.float32).copy()), dtype=np.uint8)
    except Exception:
        return dict(FAIL)

    if got_codes.shape != x.shape:
        return dict(FAIL)

    decoded_via_oracle = _decode_bits(got_codes)
    err_encode = float(np.max(np.abs(decoded_via_oracle - ref_roundtrip)))

    return {"max_abs_err": max(err_decode, err_encode)}
