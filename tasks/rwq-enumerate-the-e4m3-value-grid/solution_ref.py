import math
import numpy as np


def e4m3_value_grid() -> dict:
    finite_vals = []
    for code in range(256):
        S = (code >> 7) & 1
        E = (code >> 3) & 0xF
        M = code & 0x7

        if E == 15 and M == 7:
            continue

        sign = 1.0 if S == 0 else -1.0
        if E == 0:
            normal_val = 0.0
            subnorm = sign * math.ldexp(M / 8.0, -6)
            val = subnorm
        else:
            normal = sign * math.ldexp(1.0 + M / 8.0, E - 7)
            val = normal

        finite_vals.append(val)

    seen = set()
    unique_sorted = []
    for v in sorted(finite_vals):
        if v == 0.0:
            v = 0.0
        if v not in seen:
            seen.add(v)
            unique_sorted.append(v)

    values = np.array(unique_sorted, dtype=np.float64)

    return {
        "values": values,
        "n_finite": int(values.shape[0]),
        "max_finite": float(values[-1]) if len(values) > 0 else 0.0,
        "min_subnormal": float(math.ldexp(1.0 / 8.0, -6)),
    }
