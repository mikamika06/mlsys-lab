import numpy as np


def measure_distribution_drift(on_policy_probs, off_policy_probs):
    p = np.clip(on_policy_probs, 1e-12, 1.0)
    q = np.clip(off_policy_probs, 1e-12, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    tv_distance = 0.5 * np.sum(np.abs(p - q), axis=-1)
    kl_divergence = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    return {
        "tv_distance": float(np.mean(tv_distance)),
        "kl_divergence": float(np.mean(kl_divergence)),
        "max_tv": float(np.max(tv_distance))
    }
