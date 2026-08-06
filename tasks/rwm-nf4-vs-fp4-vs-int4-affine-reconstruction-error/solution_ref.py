import math
import numpy as np


def _nearest_reconstruct(x, codebook):
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    out = np.empty_like(x_flat)
    for i in range(x_flat.shape[0]):
        val = x_flat[i]
        best_idx = 0
        min_dist = float('inf')
        for j in range(codebook.shape[0]):
            dist = val - codebook[j]
            if dist < 0.0:
                dist = -dist
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        out[i] = codebook[best_idx]
    return out.reshape(x.shape)


def quantization_mse_triplet(W):
    w = np.asarray(W, dtype=np.float64)

    nf4 = np.array(
        [
            -1.0, -0.696192, -0.525073, -0.394917,
            -0.284441, -0.184773, -0.091050, 0.0,
            0.079580, 0.160930, 0.246112, 0.337915,
            0.440710, 0.562617, 0.722956, 1.0,
        ],
        dtype=np.float64,
    )
    fp4 = np.array(
        [
            -1.0, -0.66666667, -0.5, -0.33333333,
            -0.25, -0.16666667, -0.08333333, 0.0,
            0.08333333, 0.16666667, 0.25, 0.33333333,
            0.5, 0.66666667, 0.83333333, 1.0,
        ],
        dtype=np.float64,
    )

    w_flat = w.ravel()
    n = w_flat.shape[0]

    max_abs = 0.0
    for i in range(n):
        val = w_flat[i]
        if val < 0.0:
            val = -val
        if val > max_abs:
            max_abs = val
    scale = max_abs

    nf4_rec = _nearest_reconstruct(w / scale, nf4) * scale
    fp4_rec = _nearest_reconstruct(w / scale, fp4) * scale

    min_val = float('inf')
    max_val = float('-inf')
    for i in range(n):
        val = w_flat[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    lo = min_val
    hi = max_val
    int_scale = (hi - lo) / 15.0

    q = np.empty_like(w_flat)
    for i in range(n):
        div = (w_flat[i] - lo) / int_scale
        rounded = round(div)
        if rounded < 0.0:
            rounded = 0.0
        elif rounded > 15.0:
            rounded = 15.0
        q[i] = rounded
    q = q.reshape(w.shape)
    int_rec = q * int_scale + lo

    nf4_rec_flat = nf4_rec.ravel()
    fp4_rec_flat = fp4_rec.ravel()
    int_rec_flat = int_rec.ravel()

    sum_sq_nf4 = 0.0
    sum_sq_fp4 = 0.0
    sum_sq_int = 0.0
    for i in range(n):
        diff_nf4 = w_flat[i] - nf4_rec_flat[i]
        sum_sq_nf4 += diff_nf4 * diff_nf4

        diff_fp4 = w_flat[i] - fp4_rec_flat[i]
        sum_sq_fp4 += diff_fp4 * diff_fp4

        diff_int = w_flat[i] - int_rec_flat[i]
        sum_sq_int += diff_int * diff_int

    return (
        float(sum_sq_nf4 / n),
        float(sum_sq_fp4 / n),
        float(sum_sq_int / n),
    )
