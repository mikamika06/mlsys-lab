import math

def decode_e2m1(val: int, bias: int, has_nan: bool, has_inf: bool) -> float:
    s = (val >> 3) & 1
    e = (val >> 1) & 3
    m = val & 1
    sign = -1.0 if s == 1 else 1.0

    if e == 3:
        if has_nan and m == 1:
            return float('nan')
        if has_inf and m == 0:
            return float('inf') * sign

    if e == 0:
        if m == 0:
            return 0.0 * sign
        return sign * (2.0 ** (1 - bias)) * 0.5

    return sign * (2.0 ** (e - bias)) * (1.0 + m * 0.5)


def enumerate_values(bias: int, has_nan: bool, has_inf: bool) -> list[float]:
    return [decode_e2m1(i, bias, has_nan, has_inf) for i in range(16)]


def quantize(tensor: list[float], bias: int, has_nan: bool, has_inf: bool) -> list[float]:
    valid_vals = []
    for i in range(16):
        v = decode_e2m1(i, bias, has_nan, has_inf)
        if not math.isnan(v) and not math.isinf(v):
            valid_vals.append(v)

    out = []
    for x in tensor:
        best_v = valid_vals[0]
        best_d = abs(x - best_v)
        for v in valid_vals[1:]:
            d = abs(x - v)
            if d < best_d:
                best_d = d
                best_v = v
            elif d == best_d:
                if abs(v) > abs(best_v):
                    best_v = v
        out.append(best_v)
    return out
