import numpy as np


def entropy_calibration_threshold(tensor, num_bins=128, quant_max=127.0):
    arr = np.asarray(tensor, dtype=np.float32)
    arr = np.abs(arr)
    amax = float(np.max(arr))
    if amax == 0.0:
        return 1.0

    hist, bin_edges = np.histogram(arr, bins=num_bins, range=(0.0, amax))
    total = np.sum(hist)
    if total == 0:
        return amax / quant_max

    p = hist.astype(np.float32) / total

    best_threshold_idx = num_bins - 1
    min_kl = float("inf")

    for i in range(int(quant_max), num_bins):
        threshold = bin_edges[i + 1]
        quantized_p = np.zeros(i + 1, dtype=np.float32)

        quantized_p[:i] = p[:i]
        quantized_p[i - 1] += np.sum(p[i:])

        reference_p = p[:i + 1]
        reference_p = reference_p / np.sum(reference_p)

        expanded_p = np.zeros(i + 1, dtype=np.float32)
        expanded_p[:i] = quantized_p[:i]
        expanded_p[i] = quantized_p[i]

        sum_exp = np.sum(expanded_p)
        if sum_exp > 0:
            expanded_p /= sum_exp

        nonzero = (reference_p > 0) & (expanded_p > 0)
        if np.any(nonzero):
            kl = np.sum(reference_p[nonzero] * np.log(reference_p[nonzero] / expanded_p[nonzero]))
            if kl < min_kl:
                min_kl = kl
                best_threshold_idx = i

    return float(bin_edges[best_threshold_idx + 1] / quant_max)
