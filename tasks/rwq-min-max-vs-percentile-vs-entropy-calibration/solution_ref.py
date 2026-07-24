import numpy as np


def _entropy_threshold(x_abs: np.ndarray, num_bits: int) -> float:
    """TensorRT-style KL-divergence ("entropy") calibration threshold.

    Builds a fine-grained reference histogram of |x|, then for each candidate
    prefix length (a multiple of num_quant_bins) simulates collapsing that
    prefix down to num_quant_bins quantization bins and expanding it back,
    picking the prefix whose KL divergence to the true reference distribution
    is smallest.
    """
    num_quant_bins = 2 ** (num_bits - 1)
    num_bins = 16 * num_quant_bins

    x_max = float(x_abs.max())
    if x_max <= 0.0:
        return 1e-12

    hist, edges = np.histogram(x_abs, bins=num_bins, range=(0.0, x_max))
    bin_width = edges[1] - edges[0]
    hist = hist.astype(np.float64)

    eps = 1e-8
    best_kl = np.inf
    best_i = num_quant_bins

    for i in range(num_quant_bins, num_bins + 1, num_quant_bins):
        P = hist[:i].copy()
        P[-1] += hist[i:].sum()
        P_sum = P.sum()
        if P_sum <= 0:
            continue

        group_size = i // num_quant_bins
        Q = np.zeros(i, dtype=np.float64)
        for g in range(num_quant_bins):
            g0, g1 = g * group_size, (g + 1) * group_size
            group = P[g0:g1]
            group_sum = group.sum()
            nonzero = group > 0
            n_nonzero = int(nonzero.sum())
            if n_nonzero > 0 and group_sum > 0:
                Q[g0:g1][nonzero] = group_sum / n_nonzero

        Q_sum = Q.sum()
        if Q_sum <= 0:
            continue

        Pn = P / P_sum
        Pn = np.where(Pn == 0, eps, Pn)
        Pn = Pn / Pn.sum()

        Qn = Q / Q_sum
        Qn = np.where(Qn == 0, eps, Qn)
        Qn = Qn / Qn.sum()

        kl = float(np.sum(Pn * np.log(Pn / Qn)))
        if kl < best_kl:
            best_kl = kl
            best_i = i

    return best_i * bin_width


def _quantize_mse(x: np.ndarray, threshold: float, num_bits: int) -> float:
    """Symmetric linear quantization to num_bits, round-trip MSE."""
    levels = 2 ** (num_bits - 1) - 1
    threshold = max(threshold, 1e-12)
    scale = threshold / levels
    q = np.clip(np.round(x / scale), -levels, levels)
    x_hat = q * scale
    return float(np.mean((x - x_hat) ** 2))


def calibrate_and_score(X: np.ndarray, num_bits: int = 8) -> dict:
    """Score three int8 calibration methods on X by symmetric-quantization MSE.

    Returns {'minmax': mse, 'percentile': mse, 'entropy': mse}, each the mean
    squared reconstruction error of round-tripping X through symmetric linear
    quantization with that method's clipping threshold.
    """
    x = np.asarray(X, dtype=np.float64).ravel()
    x_abs = np.abs(x)

    t_minmax = float(x_abs.max())
    t_percentile = float(np.percentile(x_abs, 99.99))
    t_entropy = _entropy_threshold(x_abs, num_bits)

    return {
        "minmax": _quantize_mse(x, t_minmax, num_bits),
        "percentile": _quantize_mse(x, t_percentile, num_bits),
        "entropy": _quantize_mse(x, t_entropy, num_bits),
    }
