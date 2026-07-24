import numpy as np


def _e4m3_values():
    values = []
    for sign in (-1, 1):
        for exp in range(-6, 8):
            for mant in range(8):
                if exp == -6:
                    v = (2 ** -6) * (mant / 8.0)
                else:
                    v = (2 ** exp) * (1.0 + mant / 8.0)
                values.append(sign * v)
    values.extend([-448.0, 448.0])
    return np.unique(np.array(values, dtype=np.float64))


_VALUES = _e4m3_values()


def _quant_dequant(x, scale):
    y = np.clip(x / scale, -448.0, 448.0)
    result = np.empty_like(y)
    for idx, value in np.ndenumerate(y):
        result[idx] = _VALUES[np.argmin(np.abs(_VALUES - value))]
    return result * scale


def search_fp8_scale(x):
    x = np.asarray(x, dtype=np.float64)
    amax = np.max(np.abs(x))
    s0 = amax / 448.0 if amax != 0 else 1.0
    scales = np.array(
        [s0 * (2.0 ** ((i - 4) / 8.0)) for i in range(9)],
        dtype=np.float64,
    )
    mses = np.array(
        [np.mean((x - _quant_dequant(x, s)) ** 2) for s in scales],
        dtype=np.float64,
    )
    return int(np.argmin(mses)), scales, mses
