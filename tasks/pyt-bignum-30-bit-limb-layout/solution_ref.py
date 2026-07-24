import struct


def pack_bignum(x: int) -> bytes:
    bits = 30
    base_mask = (1 << bits) - 1

    sign = 1 if x < 0 else 0
    value = abs(x)

    limbs = []
    while value:
        limbs.append(value & base_mask)
        value >>= bits

    out = bytearray()
    out.append(sign)
    out.extend(struct.pack("<I", len(limbs)))

    for limb in limbs:
        out.extend(struct.pack("<I", limb))

    return bytes(out)
