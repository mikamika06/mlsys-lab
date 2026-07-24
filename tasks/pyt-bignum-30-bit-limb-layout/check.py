import struct
import sys

from mlsys import scorers


def _oracle_pack(x):
    bits = sys.int_info.bits_per_digit
    base = 1 << bits
    sign = 1 if x < 0 else 0
    value = abs(x)

    limbs = []
    while value:
        limbs.append(value & (base - 1))
        value >>= bits

    out = bytearray()
    out.append(sign)
    out.extend(struct.pack("<I", len(limbs)))
    for limb in limbs:
        out.extend(struct.pack("<I", limb))
    return bytes(out)


def grade(sol, fx) -> dict:
    cases = [
        0,
        1,
        -1,
        (1 << 30) - 1,
        1 << 30,
        (1 << 60) + 12345,
        -((1 << 90) + (1 << 31) + 7),
        (1 << 300) + (1 << 150) + 42,
    ]

    scores = []
    for x in cases:
        try:
            got = sol.pack_bignum(x)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        scores.append(scorers.byte_exact_fraction(got, _oracle_pack(x)))

    return {"byte_exact_fraction": min(scores)}
