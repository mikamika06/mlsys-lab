import math


def _e4m3_value_grid():
    """Return list of all finite E4M3 representable values."""
    vals = set()
    for bits in range(256):
        sign = -1.0 if (bits >> 7) else 1.0
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        if exp == 15 and mant == 7:
            continue
        if exp == 0:
            val = sign * (2 ** -6) * (mant / 8.0)
        else:
            val = sign * (2 ** (exp - 7)) * (1.0 + mant / 8.0)
        vals.add(val)

    vals_list = list(vals)
    n = 0
    for _ in vals_list:
        n += 1
    for i in range(n):
        for j in range(0, n - i - 1):
            if vals_list[j] > vals_list[j + 1]:
                temp = vals_list[j]
                vals_list[j] = vals_list[j + 1]
                vals_list[j + 1] = temp

    return vals_list


def _cast_single(v):
    if math.isnan(v):
        return float('nan')

    if v < -448.0:
        v_clamped = -448.0
    elif v > 448.0:
        v_clamped = 448.0
    else:
        v_clamped = float(v)

    grid = _e4m3_value_grid()
    grid_len = len(grid)

    diffs = []
    for j in range(grid_len):
        diff = grid[j] - v_clamped
        if diff < 0.0:
            diff = -diff
        diffs.append(diff)

    min_d = diffs[0]
    for j in range(1, grid_len):
        if diffs[j] < min_d:
            min_d = diffs[j]

    candidate_indices = []
    for j in range(grid_len):
        d = diffs[j] - min_d
        if d < 0:
            d = -d
        if d < 1e-14:
            candidate_indices.append(j)

    if len(candidate_indices) == 1:
        return grid[candidate_indices[0]]
    else:
        chosen = None
        for idx in candidate_indices:
            val = grid[idx]
            for bits in range(256):
                sign_b = -1.0 if (bits >> 7) else 1.0
                exp_b = (bits >> 3) & 0xF
                mant_b = bits & 0x7
                if exp_b == 15 and mant_b == 7:
                    continue
                if exp_b == 0:
                    vb = sign_b * (2 ** -6) * (mant_b / 8.0)
                else:
                    vb = sign_b * (2 ** (exp_b - 7)) * (1.0 + mant_b / 8.0)

                diff_v = vb - val
                if diff_v < 0:
                    diff_v = -diff_v
                if diff_v < 1e-15 and mant_b % 2 == 0:
                    chosen = val
                    break
            if chosen is not None:
                break
        if chosen is not None:
            return chosen
        else:
            return grid[candidate_indices[0]]


def _process_structure(x):
    if isinstance(x, (int, float)):
        return _cast_single(float(x))
    elif isinstance(x, list):
        return [_process_structure(item) for item in x]
    else:
        return [_process_structure(item) for item in x]


def cast_to_e4m3(x: list[float]) -> list[float]:
    """Cast float list to nearest E4M3 value (round-to-nearest-even, clamp at +-448)."""
    return _process_structure(x)
