import numpy as np


def search_clip_ratio(W, ratios, bits):
    W = np.asarray(W, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)

    qmax = (1 << (bits - 1)) - 1
    max_abs = np.max(np.abs(W), axis=1)

    mse_curve = []

    for ratio in ratios:
        bounds = (max_abs * ratio)[:, None]
        scales = bounds / qmax
        clipped = np.clip(W, -bounds, bounds)
        quantized = np.round(clipped / scales)
        reconstructed = quantized * scales
        mse_curve.append(np.mean((W - reconstructed) ** 2))

    mse_curve = np.asarray(mse_curve, dtype=np.float64)
    return int(np.argmin(mse_curve)), mse_curve
