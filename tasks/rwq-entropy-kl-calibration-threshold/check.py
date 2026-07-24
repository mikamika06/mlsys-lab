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


def _oracle(activations, num_bins, candidate_indices):
    edges = np.linspace(0.0, float(np.max(activations)), num_bins + 1)
    hist, _ = np.histogram(activations, bins=edges)
    best_idx = None
    best_kl = None
    for k in candidate_indices:
        kl = _kl_for_threshold(activations, edges, hist, int(k))
        if best_kl is None or kl < best_kl:
            best_kl = kl
            best_idx = int(k)
    return best_idx, float(best_kl)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.02, 0.1, 0.12, 0.3, 0.8, 1.2, 2.0, 2.1], dtype=np.float64),
            16,
            np.array([4, 8, 12], dtype=np.int64),
        ),
        (
            np.array([0.0, 0.05, 0.07, 0.2, 0.25, 0.4, 0.9, 1.8, 3.0], dtype=np.float64),
            24,
            np.array([6, 10, 16, 20], dtype=np.int64),
        ),
        (
            np.array([0.1, 0.15, 0.2, 0.21, 0.22, 0.7, 1.4, 1.5, 1.6, 3.2], dtype=np.float64),
            20,
            np.array([5, 9, 13, 17], dtype=np.int64),
        ),
    ]

    exact = 1.0
    err = 0.0
    for activations, bins, candidates in cases:
        ref_idx, ref_kl = _oracle(activations, bins, candidates)
        try:
            got_idx = int(sol.calibrate_threshold_index(
                activations.copy(), bins, candidates.copy()
            ))
        except Exception:
            return {"argmin_index": 0.0, "kl_error": float("inf")}

        if got_idx != ref_idx:
            exact = 0.0

        try:
            edges = np.linspace(0.0, float(np.max(activations)), bins + 1)
            hist, _ = np.histogram(activations, bins=edges)
            got_kl = _kl_for_threshold(activations, edges, hist, got_idx)
            err = max(err, abs(got_kl - ref_kl))
        except Exception:
            err = float("inf")

    return {"argmin_index": exact, "kl_error": err}
