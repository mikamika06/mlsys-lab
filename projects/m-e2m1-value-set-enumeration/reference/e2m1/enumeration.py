def enumerate_e2m1():
    values = []
    for bits in range(16):
        s = (bits >> 3) & 1
        e = (bits >> 1) & 3
        m = bits & 1
        sign = -1.0 if s else 1.0
        if e == 0:
            if m == 0:
                val = 0.0 if s == 0 else -0.0
            else:
                val = sign * (2.0 ** (1 - 1)) * (m / 2.0)
        else:
            val = sign * (2.0 ** (e - 1)) * (1.0 + m / 2.0)
        values.append({"bits": bits, "value": float(val)})
    return values
