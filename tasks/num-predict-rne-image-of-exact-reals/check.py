"""Grader for `num-predict-rne-image-of-exact-reals`.

Oracle: a pure-integer round-to-nearest-even implementation, operating
directly on the exact (num, den) pair with Python's arbitrary-precision
integers. It never goes through an intermediate float64 and never calls
`sol.rne_fp32_bits` — fully independent of the candidate.
"""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np


def _ge_pow2(num: int, den: int, e: int) -> bool:
    """True iff num/den >= 2**e, using only non-negative shifts."""
    if e >= 0:
        return num >= (den << e)
    return (num << (-e)) >= den


def _oracle_bits(num: int, den: int) -> int:
    """Exact RNE rounding of num/den to a float32 bit pattern."""
    assert den > 0
    sign = 0
    if num < 0:
        sign = 1
        num = -num
    if num == 0:
        return sign << 31

    # unbiased exponent e such that 2**e <= num/den < 2**(e+1)
    e = num.bit_length() - den.bit_length()
    while _ge_pow2(num, den, e + 1):
        e += 1
    while not _ge_pow2(num, den, e):
        e -= 1

    # 24-bit significand target = (num/den) * 2**(23-e), rounded RNE
    shift = 23 - e
    if shift >= 0:
        numerator, denom = num << shift, den
    else:
        numerator, denom = num, den << (-shift)
    q, r = divmod(numerator, denom)

    twice_r = 2 * r
    if twice_r > denom or (twice_r == denom and (q & 1) == 1):
        q += 1
    if q >= (1 << 24):  # mantissa overflow -> carry into exponent
        q >>= 1
        e += 1

    mantissa = q - (1 << 23)
    exp_field = e + 127
    return (sign << 31) | (exp_field << 23) | mantissa


def _dr_trap_case(e: int, m: int, sign: int) -> tuple[int, int]:
    """A (num, den) pair engineered so the exact value sits just below a
    float32 tie point, but rounding through float64 first lands exactly on
    the tie and then ties-to-even the wrong way -- a double-rounding trap.
    """
    H = Fraction(2 ** 24 + 2 * m + 1, 2 ** 24) * Fraction(2) ** e
    delta = Fraction(2) ** (e - 60)
    v = H - delta
    if sign < 0:
        v = -v
    return v.numerator, v.denominator


def _tie_case(e: int, m: int, sign: int) -> tuple[int, int]:
    """A (num, den) pair that is an EXACT float32-level tie between
    mantissa m and m+1 at exponent e."""
    v = Fraction(2 ** 24 + 2 * m + 1, 2 ** 24) * Fraction(2) ** e
    if sign < 0:
        v = -v
    return v.numerator, v.denominator


def _build_cases() -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = [(0, 1)]

    rng = np.random.default_rng(0)
    for _ in range(2000):
        num = int(rng.integers(1, 10_000_000))
        den = int(rng.integers(1, 10_000_000))
        if rng.random() < 0.5:
            num = -num
        cases.append((num, den))

    rnd = random.Random(12345)
    for _ in range(500):
        nbits = rnd.randint(60, 200)
        dbits = rnd.randint(60, 200)
        num = rnd.getrandbits(nbits) | 1
        den = rnd.getrandbits(dbits) | 1
        if rnd.random() < 0.5:
            num = -num
        cases.append((num, den))

    for e in (-40, -5, 0, 5, 40):
        for m in (1, 3, 8_388_605):
            for sign in (1, -1):
                cases.append(_dr_trap_case(e, m, sign))
                cases.append(_tie_case(e, m, sign))
    # tie that overflows the mantissa on round-up (m odd, m+1 == 2**23)
    for e in (-10, 0, 17):
        for sign in (1, -1):
            cases.append(_tie_case(e, (1 << 23) - 1, sign))

    return cases


def grade(sol, fx) -> dict:
    cases = _build_cases()
    ref = [_oracle_bits(num, den) for num, den in cases]

    try:
        got = sol.rne_fp32_bits(list(cases))
    except Exception:
        return {"exact_match": 0.0}

    try:
        got = list(got)
    except Exception:
        return {"exact_match": 0.0}

    if len(got) != len(ref):
        return {"exact_match": 0.0}

    matches = 0
    for g, r in zip(got, ref):
        try:
            if int(g) == int(r):
                matches += 1
        except Exception:
            pass

    return {"exact_match": matches / len(ref)}
