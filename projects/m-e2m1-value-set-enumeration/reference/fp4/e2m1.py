def enumerate_e2m1():
    values = []
    for bits in range(16):
        sign = (bits >> 3) & 1
        exp = (bits >> 1) & 3
        mant = bits & 1
        if exp == 0:
            val = ((-1.0) ** sign) * (mant / 2.0) * (2.0 ** (-0))
        else:
            val = ((-1.0) ** sign) * (1.0 + mant / 2.0) * (2.0 ** (exp - 1))
        values.append({"bits": bits, "sign": sign, "exp": exp, "mant": mant, "value": float(val)})
    return values


def quantize_e2m1(x):
    evs = enumerate_e2m1()
    best_bits = 0
    best_diff = float("inf")
    for item in evs:
        diff = abs(item["value"] - x)
        if diff < best_diff:
            best_diff = diff
            best_bits = item["bits"]
    return best_bits
