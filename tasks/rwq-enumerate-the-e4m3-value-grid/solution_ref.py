import numpy as np


def e4m3_value_grid() -> dict:
    codes = np.arange(256, dtype=np.int64)
    S = (codes >> 7) & 1
    E = (codes >> 3) & 0xF
    M = codes & 0x7

    sign = np.where(S == 0, 1.0, -1.0)
    normal = sign * np.ldexp(1.0 + M / 8.0, (E - 7).astype(np.int64))
    subnorm = sign * np.ldexp(M / 8.0, -6)
    val = np.where(E == 0, subnorm, normal)

    is_nan = (E == 15) & (M == 7)
    finite = val[~is_nan]
    values = np.unique(finite)

    return {
        "values": values,
        "n_finite": int(values.shape[0]),
        "max_finite": float(np.max(values)),
        "min_subnormal": float(np.ldexp(1.0 / 8.0, -6)),
    }
