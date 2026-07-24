"""Reference: exact-rational -> float32 RNE bit pattern, pure integer
arithmetic (no intermediate float64, so no double-rounding)."""
from __future__ import annotations


def _ge_pow2(num: int, den: int, e: int) -> bool:
    """True iff num/den >= 2**e, using only non-negative shifts."""
    if e >= 0:
        return num >= (den << e)
    return (num << (-e)) >= den


def _round_one(num: int, den: int) -> int:
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

    # target 24-bit significand = (num/den) * 2**(23-e), rounded RNE
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


def rne_fp32_bits(pairs: list[tuple[int, int]]) -> list[int]:
    return [_round_one(num, den) for num, den in pairs]
