import numpy as np


def quantize_dequant_rtn_v0(W: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group asymmetric affine 4-bit round-to-nearest quantization — the
    V=0 (no learned rounding perturbation) baseline that AutoRound-style
    optimizers start from.

    W is raveled in row-major order and split into consecutive groups of
    `group_size` elements (the last group may be shorter). For each group:

        scale = (max - min) / 15          (1.0 if the group is constant)
        zero  = clip(round(-min / scale), 0, 15)
        code  = clip(round(x / scale) + zero, 0, 15)   for each x in the group

    Returns (codes, W_dq):
      codes -- uint8 array, same shape as W, values in [0, 15]
      W_dq  -- float64 array, same shape as W, dequantized reconstruction
    """
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    codes = np.empty(n, dtype=np.uint8)
    dq = np.empty(n, dtype=np.float64)

    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g[0])
        gmin = float(g[0])
        for i in range(len(g)):
            val = float(g[i])
            if val > gmax:
                gmax = val
            if val < gmin:
                gmin = val
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        raw_zero = round(-gmin / scale)
        if raw_zero < 0:
            zero = 0.0
        elif raw_zero > 15:
            zero = 15.0
        else:
            zero = float(raw_zero)
        for i in range(len(g)):
            x = float(g[i])
            r = round(x / scale)
            val_code = r + zero
            if val_code < 0:
                c = 0
            elif val_code > 15:
                c = 15
            else:
                c = int(val_code)
            idx = start + i
            codes[idx] = c
            dq[idx] = (float(c) - zero) * scale

    return codes.reshape(shape), dq.reshape(shape)
