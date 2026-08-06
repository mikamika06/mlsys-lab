def enumerate_values():
    out = []
    for bits in range(16):
        sign = (bits >> 3) & 1
        exp = (bits >> 1) & 3
        mant = bits & 1
        if exp == 0:
            val = ((-1.0) ** sign) * (mant / 2.0) * (2.0 ** 0)
        else:
            val = ((-1.0) ** sign) * (1.0 + mant / 2.0) * (2.0 ** (exp - 1))
        out.append({"bits": bits, "value": float(val)})
    return out
