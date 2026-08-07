def make_qx_quants(
    x: list[float], w: list[float], nmax: int
) -> tuple[int, list[int]]:
    n = len(x)

    amax = 0.0
    for i in range(n):
        val = x[i]
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val

    if amax == 0:
        return -1, [0] * n

    base_scale = amax / nmax
    best_idx = -1
    best_err = None
    best_codes = None

    for k in range(-15, 16):
        idx = k + 15
        scale = base_scale * (1.0 + k / 32.0)
        if scale == 0:
            continue

        codes_list = []
        err = 0.0
        for i in range(n):
            xi = x[i]
            wi = w[i]
            r = round(xi / scale)
            if r < -nmax:
                c = -nmax
            elif r > nmax:
                c = nmax
            else:
                c = int(r)
            codes_list.append(c)
            diff = xi - scale * float(c)
            err += wi * (diff * diff)

        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
            best_codes = codes_list

    return best_idx, best_codes
