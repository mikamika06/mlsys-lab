import math
import numpy as np


def search_clip_ratio(W, ratios, bits):
    W = np.asarray(W, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)

    qmax = (1 << (bits - 1)) - 1
    
    nrows = W.shape[0]
    ncols = W.shape[1]

    max_abs = [0.0] * nrows
    for i in range(nrows):
        m = 0.0
        for j in range(ncols):
            val = W[i, j]
            if val < 0.0:
                val = -val
            if val > m:
                m = val
        max_abs[i] = m

    mse_curve = []

    for ratio in ratios:
        bounds = [m * ratio for m in max_abs]
        scales = [b / qmax for b in bounds]

        total_sq_err = 0.0
        count = 0

        for i in range(nrows):
            b = bounds[i]
            s = scales[i]
            for j in range(ncols):
                val = W[i, j]
                if val < -b:
                    clipped = -b
                elif val > b:
                    clipped = b
                else:
                    clipped = val

                quantized = round(clipped / s)
                reconstructed = quantized * s
                diff = val - reconstructed
                total_sq_err += diff * diff
                count += 1

        mse_curve.append(total_sq_err / count)

    mse_curve_arr = np.asarray(mse_curve, dtype=np.float64)

    min_val = mse_curve_arr[0]
    best_idx = 0
    for idx in range(1, len(mse_curve_arr)):
        val = mse_curve_arr[idx]
        if val < min_val:
            min_val = val
            best_idx = idx

    return int(best_idx), mse_curve_arr
