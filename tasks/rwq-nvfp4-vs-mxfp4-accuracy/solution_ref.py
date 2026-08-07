import math


_CODEBOOK = [
    -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0
]


def _quantize(x, block_size, pow2_scale):
    out = [0.0] * len(x)
    n = len(x)
    max_codebook_val = 1.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        max_abs = 0.0
        for i in range(start, end):
            val = x[i]
            if val < 0:
                val = -val
            if val > max_abs:
                max_abs = val

        scale = max_abs / max_codebook_val

        if scale == 0:
            q_scale = 1.0
        elif pow2_scale:
            q_scale = 2.0 ** math.ceil(math.log2(scale))
        else:
            q_scale = scale

        for i in range(start, end):
            val = x[i]
            best_idx = 0
            min_diff = float("inf")
            for c_idx in range(len(_CODEBOOK)):
                diff = (val / q_scale) - _CODEBOOK[c_idx]
                if diff < 0:
                    diff = -diff
                if diff < min_diff:
                    min_diff = diff
                    best_idx = c_idx
            out[i] = q_scale * _CODEBOOK[best_idx]

    return out


def fp4_accuracy_comparison(weight):
    nv = _quantize(weight, 16, False)
    mx = _quantize(weight, 32, True)

    n = len(weight)
    sum_sq_nv = 0.0
    sum_sq_mx = 0.0

    for i in range(n):
        w_val = float(weight[i])
        diff_nv = w_val - nv[i]
        diff_mx = w_val - mx[i]
        sum_sq_nv += diff_nv * diff_nv
        sum_sq_mx += diff_mx * diff_mx

    nv_rmse = float(math.sqrt(sum_sq_nv / n))
    mx_rmse = float(math.sqrt(sum_sq_mx / n))

    return nv_rmse, mx_rmse
