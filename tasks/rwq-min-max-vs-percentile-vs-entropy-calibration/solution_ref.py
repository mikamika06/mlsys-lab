import math
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

    x_max = float("-inf")
    for val in x_abs:
        if val > x_max:
            x_max = float(val)
    if x_max <= 0.0:
        return 1e-12

    bin_width = x_max / num_bins
    hist_list = [0.0] * num_bins
    for val in x_abs:
        if val < 0.0:
            continue
        b = int(val / bin_width)
        if b >= num_bins:
            b = num_bins - 1
        hist_list[b] += 1.0
    hist = np.array(hist_list, dtype=np.float64)

    eps = 1e-8
    best_kl = float("inf")
    best_i = num_quant_bins

    for i in range(num_quant_bins, num_bins + 1, num_quant_bins):
        P = hist[:i].copy()
        tail_sum = 0.0
        for idx in range(i, len(hist)):
            tail_sum += hist[idx]
        P[-1] += tail_sum

        P_sum = 0.0
        for val in P:
            P_sum += val
        if P_sum <= 0:
            continue

        group_size = i // num_quant_bins
        Q = np.zeros(i, dtype=np.float64)
        for g in range(num_quant_bins):
            g0, g1 = g * group_size, (g + 1) * group_size
            group = P[g0:g1]
            group_sum = 0.0
            for val in group:
                group_sum += val
            
            n_nonzero = 0
            for val in group:
                if val > 0:
                    n_nonzero += 1
            
            if n_nonzero > 0 and group_sum > 0:
                fill_val = group_sum / n_nonzero
                for idx in range(g0, g1):
                    if P[idx] > 0:
                        Q[idx] = fill_val

        Q_sum = 0.0
        for val in Q:
            Q_sum += val
        if Q_sum <= 0:
            continue

        Pn = np.zeros(i, dtype=np.float64)
        for idx in range(i):
            val = P[idx] / P_sum
            if val == 0:
                val = eps
            Pn[idx] = val
        Pn_sum = 0.0
        for val in Pn:
            Pn_sum += val
        for idx in range(i):
            Pn[idx] /= Pn_sum

        Qn = np.zeros(i, dtype=np.float64)
        for idx in range(i):
            val = Q[idx] / Q_sum
            if val == 0:
                val = eps
            Qn[idx] = val
        Qn_sum = 0.0
        for val in Qn:
            Qn_sum += val
        for idx in range(i):
            Qn[idx] /= Qn_sum

        kl = 0.0
        for idx in range(i):
            kl += Pn[idx] * math.log(Pn[idx] / Qn[idx])
        kl = float(kl)

        if kl < best_kl:
            best_kl = kl
            best_i = i

    return best_i * bin_width


def _quantize_mse(x: np.ndarray, threshold: float, num_bits: int) -> float:
    """Symmetric linear quantization to num_bits, round-trip MSE."""
    levels = 2 ** (num_bits - 1) - 1
    threshold = max(threshold, 1e-12)
    scale = threshold / levels
    
    q = np.zeros(x.shape, dtype=np.float64)
    for idx in range(len(x)):
        val = x[idx] / scale
        rounded = round(val)
        if rounded < -levels:
            rounded = -levels
        elif rounded > levels:
            rounded = levels
        q[idx] = rounded

    x_hat = q * scale

    sq_err_sum = 0.0
    for idx in range(len(x)):
        diff = x[idx] - x_hat[idx]
        sq_err_sum += diff * diff
    return float(sq_err_sum / len(x))


def calibrate_and_score(X: np.ndarray, num_bits: int = 8) -> dict:
    """Score three int8 calibration methods on X by symmetric-quantization MSE.

    Returns {'minmax': mse, 'percentile': mse, 'entropy': mse}, each the mean
    squared reconstruction error of round-tripping X through symmetric linear
    quantization with that method's clipping threshold.
    """
    x = np.asarray(X, dtype=np.float64).ravel()
    x_abs = np.zeros(x.shape, dtype=np.float64)
    for i in range(len(x)):
        val = x[i]
        x_abs[i] = val if val >= 0.0 else -val

    t_minmax = float("-inf")
    for val in x_abs:
        if val > t_minmax:
            t_minmax = float(val)

    x_abs_list = sorted(x_abs.tolist())
    n_elem = len(x_abs_list)
    index = (99.99 / 100.0) * (n_elem - 1)
    i_idx = int(index)
    frac = index - i_idx
    if i_idx + 1 < n_elem:
        t_percentile = float(x_abs_list[i_idx] + frac * (x_abs_list[i_idx + 1] - x_abs_list[i_idx]))
    else:
        t_percentile = float(x_abs_list[i_idx])

    t_entropy = _entropy_threshold(x_abs, num_bits)

    return {
        "minmax": _quantize_mse(x, t_minmax, num_bits),
        "percentile": _quantize_mse(x, t_percentile, num_bits),
        "entropy": _quantize_mse(x, t_entropy, num_bits),
    }
