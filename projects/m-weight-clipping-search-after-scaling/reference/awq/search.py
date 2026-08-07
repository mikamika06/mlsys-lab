import numpy as np


def compute_activation_scales(act_means: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    s = np.power(np.maximum(act_means, 1e-8), gamma)
    mean_s = np.mean(s)
    if mean_s == 0:
        return np.ones_like(act_means)
    return s / mean_s


def quantize_asym_int4(w: np.ndarray, clip_ratio: float) -> np.ndarray:
    max_val = np.max(np.abs(w), axis=-1, keepdims=True) * clip_ratio
    max_val = np.maximum(max_val, 1e-8)
    w_clipped = np.clip(w, -max_val, max_val)
    qmin, qmax = -8, 7
    scale = (2 * max_val) / (qmax - qmin)
    q = np.round(w_clipped / scale)
    q = np.clip(q, qmin, qmax)
    return q * scale


def search_weight_clipping(w_scaled: np.ndarray, x: np.ndarray, n_grid: int = 10, min_clip: float = 0.4) -> np.ndarray:
    out_features, in_features = w_scaled.shape
    ratios = np.linspace(min_clip, 1.0, n_grid)
    best_indices = np.zeros(out_features, dtype=int)

    y_true = x @ w_scaled.T

    for i in range(out_features):
        w_channel = w_scaled[i:i+1, :]
        best_err = float("inf")
        best_idx = n_grid - 1

        for idx, r in enumerate(ratios):
            w_q = quantize_asym_int4(w_channel, r)
            y_pred = x @ w_q.T
            err = np.mean((y_true[:, i:i+1] - y_pred) ** 2)
            if err < best_err:
                best_err = err
                best_idx = idx
        best_indices[i] = best_idx

    return best_indices
