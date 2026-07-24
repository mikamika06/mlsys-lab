from fractions import Fraction

import numpy as np

_FAIL = {
    "valid_input": 0.0,
    "paths_differ": 0.0,
    "direct_match": 0.0,
    "double_match": 0.0,
}


def _round_rne(x: Fraction, frac_bits: int) -> int:
    """Round x in [1,2) to the grid of spacing 2**-frac_bits, ties to even.

    Returns the integer n with rounded value == n / 2**frac_bits.
    Pure exact integer arithmetic — no float is involved anywhere.
    """
    scaled = x * (1 << frac_bits)          # exact Fraction
    n, r = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * r
    if twice > scaled.denominator:
        n += 1
    elif twice == scaled.denominator and (n & 1):
        n += 1                              # tie -> even
    return int(n)


def _bits32(v) -> int:
    return int(np.float32(v).view(np.uint32))


def grade(sol, fx) -> dict:
    try:
        out = sol.double_rounding_counterexample()
        num, den, direct, doubled = out
        num = int(num)
        den = int(den)
    except Exception:
        return dict(_FAIL)

    # --- valid_input -------------------------------------------------------
    if den <= 0 or (den & (den - 1)) != 0:
        return dict(_FAIL)
    x = Fraction(num, den)
    if not (Fraction(1) <= x < Fraction(2)):
        return dict(_FAIL)
    res = {"valid_input": 1.0, "paths_differ": 0.0,
           "direct_match": 0.0, "double_match": 0.0}

    # --- reference path A: exact rational -> binary32 (RNE) ----------------
    n32 = _round_rne(x, 23)
    # n32 <= 2**24 -> exact in float32, and 2**-23 is exact
    ref_direct = np.float32(n32) * np.float32(2.0 ** -23)

    # --- reference path B: exact rational -> binary64 (RNE) -> binary32 ----
    n64 = _round_rne(x, 52)
    # n64 <= 2**53 -> exact in float64, and 2**-52 is exact
    ref_64 = np.float64(n64) * np.float64(2.0 ** -52)
    ref_double = np.float32(ref_64)          # NumPy's real IEEE narrowing

    b_direct = int(ref_direct.view(np.uint32))
    b_double = int(ref_double.view(np.uint32))
    res["paths_differ"] = 1.0 if b_direct != b_double else 0.0

    # --- learner's claimed roundings, compared bit-for-bit ------------------
    try:
        res["direct_match"] = 1.0 if _bits32(direct) == b_direct else 0.0
        res["double_match"] = 1.0 if _bits32(doubled) == b_double else 0.0
    except Exception:
        res["direct_match"] = 0.0
        res["double_match"] = 0.0

    return res
