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
    return np.array(sorted(list(set(values))), dtype=np.float64)


_VALUES = _e4m3_values()


def _quant_dequant(x, scale):
    result = np.empty_like(x, dtype=np.float64)
    for idx, val in np.ndenumerate(x):
        clipped_val = val / scale
        if clipped_val < -448.0:
            clipped_val = -448.0
        elif clipped_val > 448.0:
            clipped_val = 448.0
        
        min_diff = float("inf")
        best_v = _VALUES[0]
        for v in _VALUES:
            diff = v - clipped_val
            if diff < 0.0:
                diff = -diff
            if diff < min_diff:
                min_diff = diff
                best_v = v
        result[idx] = best_v
    return result * scale


def search_fp8_scale(x):
    x = np.asarray(x, dtype=np.float64)
    amax = 0.0
    for idx, val in np.ndenumerate(x):
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
    s0 = amax / 448.0 if amax != 0.0 else 1.0
    scales = np.array(
        [s0 * (2.0 ** ((i - 4) / 8.0)) for i in range(9)],
        dtype=np.float64,
    )
    mses_list = []
    for s in scales:
        qd = _quant_dequant(x, s)
        total_sq_diff = 0.0
        count = 0
        for idx, val in np.ndenumerate(x):
            diff = val - qd[idx]
            total_sq_diff += diff * diff
            count += 1
        mses_list.append(total_sq_diff / count)
    mses = np.array(mses_list, dtype=np.float64)
    
    min_mse = mses[0]
    best_i = 0
    for i in range(len(mses)):
        if mses[i] < min_mse:
            min_mse = mses[i]
            best_i = i
            
    return int(best_i), scales, mses
