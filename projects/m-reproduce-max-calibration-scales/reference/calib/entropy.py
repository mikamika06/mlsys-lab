import numpy as np


def compute_entropy_scale(tensor, num_bins=128, qmax=127.0):
    data = np.abs(tensor)
    max_val = np.max(data)
    if max_val == 0:
        return 1.0
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(0, max_val))
    bin_width = bin_edges[1] - bin_edges[0]
    best_threshold_idx = 0
    min_kl = float("inf")
    for i in range(10, num_bins):
        threshold_idx = i
        quantized_bins = np.zeros(int(qmax) + 1)
        for j in range(threshold_idx):
            q_idx = int(j * qmax / threshold_idx)
            quantized_bins[q_idx] += hist[j]
        overflow = np.sum(hist[threshold_idx:])
        quantized_bins[int(qmax)] += overflow
        reference_distribution = np.zeros(threshold_idx)
        for j in range(threshold_idx):
            q_idx = int(j * qmax / threshold_idx)
            cnt = sum(1 for k in range(threshold_idx) if int(k * qmax / threshold_idx) == q_idx)
            reference_distribution[j] = quantized_bins[q_idx] / max(1, cnt)
        P = hist[:threshold_idx].astype(np.float64) + 1e-12
        P /= np.sum(P)
        Q = reference_distribution + 1e-12
        Q /= np.sum(Q)
        kl = np.sum(P * np.log(P / Q))
        if kl < min_kl:
            min_kl = kl
            best_threshold_idx = threshold_idx
    return float((best_threshold_idx + 0.5) * bin_width / qmax)
