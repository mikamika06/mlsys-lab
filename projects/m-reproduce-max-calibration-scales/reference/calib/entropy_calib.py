import numpy as np


def compute_entropy_scale(
    tensor: np.ndarray,
    num_bins: int = 2048,
    num_quant_steps: int = 128,
    max_bound: float = 127.0,
) -> float:
    abs_tensor = np.abs(tensor.astype(np.float64)).flatten()
    max_val = float(np.max(abs_tensor)) if abs_tensor.size > 0 else 0.0
    if max_val == 0.0:
        return 1.0

    hist, bin_edges = np.histogram(abs_tensor, bins=num_bins, range=(0.0, max_val))
    hist = hist.astype(np.float64)

    zero_bin = 0
    best_kl = float("inf")
    best_threshold = max_val

    total_elements = np.sum(hist)
    if total_elements == 0:
        return 1.0

    start_bin = max(1, num_quant_steps)

    for threshold_bin in range(start_bin, num_bins + 1):
        p_hist = hist[:threshold_bin].copy()
        outliers = np.sum(hist[threshold_bin:])
        p_hist[-1] += outliers

        if np.sum(p_hist) == 0:
            continue

        p = p_hist / np.sum(p_hist)

        quant_bin_size = threshold_bin / float(num_quant_steps)
        q = np.zeros(threshold_bin, dtype=np.float64)

        for i in range(num_quant_steps):
            start = int(np.floor(i * quant_bin_size))
            end = int(np.ceil((i + 1) * quant_bin_size))
            end = min(end, threshold_bin)
            if start < end:
                q[start:end] += np.sum(p[start:end])

        for i in range(num_quant_steps):
            start = int(np.floor(i * quant_bin_size))
            end = int(np.ceil((i + 1) * quant_bin_size))
            end = min(end, threshold_bin)
            count = end - start
            if count > 0 and q[start] > 0:
                q[start:end] = q[start] / count

        eps = 1e-12
        p = np.where(p == 0, eps, p)
        q = np.where(q == 0, eps, q)

        p = p / np.sum(p)
        q = q / np.sum(q)

        kl_div = np.sum(p * np.log(p / q))

        if kl_div < best_kl:
            best_kl = kl_div
            best_threshold = bin_edges[threshold_bin]

    return float(best_threshold / max_bound)
