import math
import struct

BIAS = 1023
MANT_BITS = 52
MANT_MASK = (1 << MANT_BITS) - 1


def float_fields(x: float) -> tuple[int, int, int]:
    """(sign_bit, biased_exponent, stored_mantissa) of the IEEE-754 binary64 pattern."""
    bits = struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    sign = (bits >> 63) & 1
    exponent = (bits >> MANT_BITS) & 0x7FF
    mantissa = bits & MANT_MASK
    return int(sign), int(exponent), int(mantissa)


def exact_ratio(x: float) -> tuple[int, int]:
    """Exact value of the float as numerator/denominator in lowest terms."""
    sign, exponent, mantissa = float_fields(x)

    if exponent == 0:                       # subnormal (or zero): no implicit leading 1
        significand = mantissa
        exp2 = 1 - BIAS - MANT_BITS         # = -1074
    else:                                   # normal: implicit leading 1
        significand = mantissa | (1 << MANT_BITS)
        exp2 = exponent - BIAS - MANT_BITS

    if exp2 >= 0:
        num, den = significand << exp2, 1
    else:
        num, den = significand, 1 << (-exp2)

    g = math.gcd(num, den)
    if g > 1:
        num //= g
        den //= g
    if num == 0:
        den = 1
    if sign:
        num = -num
    return int(num), int(den)
