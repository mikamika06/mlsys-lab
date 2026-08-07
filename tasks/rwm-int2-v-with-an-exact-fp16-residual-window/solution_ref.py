import math


def kv_int2_residual_window(V: list[list[float]], group_size: int = 32, residual_window: int = 16) -> list[list[float]]:
    """
    Quantize all but the last `residual_window` rows of `V` to 2 bits/element
    using grouped affine (zero-point) quantization along the channel axis;
    leave the last `residual_window` rows exact. Returns the reconstructed
    (T, d) list.
    """
    T = len(V)
    d = len(V[0])
    Tq = T - residual_window

    Vq = V[:Tq]
    Vr = V[Tq:]

    ng = d // group_size

    lo = [[0.0] * ng for _ in range(Tq)]
    scale = [[0.0] * ng for _ in range(Tq)]
    Vq_hat = [[0.0] * d for _ in range(Tq)]

    for i in range(Tq):
        for g in range(ng):
            start_col = g * group_size
            min_val = Vq[i][start_col]
            max_val = Vq[i][start_col]
            for j in range(1, group_size):
                val = Vq[i][start_col + j]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
            lo[i][g] = min_val
            s = (max_val - min_val) / 3.0
            if s == 0.0:
                s = 1.0
            scale[i][g] = s

    for i in range(Tq):
        for g in range(ng):
            start_col = g * group_size
            s = scale[i][g]
            l = lo[i][g]
            for j in range(group_size):
                col = start_col + j
                c = round((Vq[i][col] - l) / s)
                if c < 0.0:
                    c = 0.0
                elif c > 3.0:
                    c = 3.0
                Vq_hat[i][col] = c * s + l

    return [row[:] for row in Vq_hat] + [row[:] for row in Vr]
