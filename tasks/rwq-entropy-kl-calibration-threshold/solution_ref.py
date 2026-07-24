import numpy as np


def _kl_for_threshold(activations, edges, hist, k):
    threshold = edges[k]
    clipped = np.minimum(activations, threshold)
    levels = np.floor(clipped / threshold * k).astype(np.int64)
    levels = np.minimum(levels, k - 1)
    reconstructed = (levels.astype(np.float64) + 0.5) / k * threshold
    qhist, _ = np.histogram(reconstructed, bins=edges)

    p = hist.astype(np.float64)
    p = p / np.sum(p)
    q = qhist.astype(np.float64)
    q = q / np.sum(q)
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def calibrate_threshold_index(activations, num_bins, candidate_indices):
    edges = np.linspace(0.0, float(np.max(activations)), num_bins + 1)
    hist, _ = np.histogram(activations, bins=edges)

    best_index = int(candidate_indices[0])
    best_kl = _kl_for_threshold(
        activations, edges, hist, best_index
    )

    for k in candidate_indices[1:]:
        k = int(k)
        kl = _kl_for_threshold(activations, edges, hist, k)
        if kl < best_kl:
            best_kl = kl
            best_index = k

    return best_index
