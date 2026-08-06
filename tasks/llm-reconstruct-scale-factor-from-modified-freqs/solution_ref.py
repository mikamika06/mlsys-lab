import numpy as np

def recover_scale_factor(orig_freqs: np.ndarray,
                         mod_freqs: np.ndarray) -> float:
    """
    Recover the scaling factor s such that mod_freqs ≈ orig_freqs * s.
    The factor is computed as the mean of element‑wise ratios, which is
    accurate when all elements are scaled by a common constant up to small noise.
    """
    orig = np.asarray(orig_freqs, dtype=np.float64)
    mod  = np.asarray(mod_freqs, dtype=np.float64)
    n = orig.shape[0]
    total = 0.0
    for i in range(n):
        total += mod[i] / orig[i]
    return float(total / n)
