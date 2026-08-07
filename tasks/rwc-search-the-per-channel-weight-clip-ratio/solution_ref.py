import math


def search_clip_ratio(W: list[list[float]], ratios: list[float], bits: int) -> tuple[int, list[float]]:
    qmax = (1 << (bits - 1)) - 1

    nrows = len(W)
    ncols = len(W[0]) if nrows > 0 else 0

    max_abs = [0.0] * nrows
    for i in range(nrows):
        m = 0.0
        for j in range(ncols):
            val = W[i][j]
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
                val = W[i][j]
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

    min_val = mse_curve[0]
    best_idx = 0
    for idx in range(1, len(mse_curve)):
        val = mse_curve[idx]
        if val < min_val:
            min_val = val
            best_idx = idx

    return int(best_idx), mse_curve
