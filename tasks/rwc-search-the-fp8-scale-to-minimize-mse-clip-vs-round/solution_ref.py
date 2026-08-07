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
    return sorted(list(set(values)))


_VALUES = _e4m3_values()


def _quant_dequant(x, scale):
    result = []
    for val in x:
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
        result.append(best_v * scale)
    return result


def search_fp8_scale(x: list[float]) -> tuple[int, list[float], list[float]]:
    amax = 0.0
    for val in x:
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
    s0 = amax / 448.0 if amax != 0.0 else 1.0
    scales = [s0 * (2.0 ** ((i - 4) / 8.0)) for i in range(9)]
    mses_list = []
    for s in scales:
        qd = _quant_dequant(x, s)
        total_sq_diff = 0.0
        count = 0
        for val, qd_val in zip(x, qd):
            diff = val - qd_val
            total_sq_diff += diff * diff
            count += 1
        mses_list.append(total_sq_diff / count)
    mses = mses_list

    min_mse = mses[0]
    best_i = 0
    for i in range(len(mses)):
        if mses[i] < min_mse:
            min_mse = mses[i]
            best_i = i

    return int(best_i), scales, mses
