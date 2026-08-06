import numpy as np


def measure_distribution_drift(on_policy_probs, off_policy_probs):
    on_p = np.array(on_policy_probs, dtype=np.float64)
    off_p = np.array(off_policy_probs, dtype=np.float64)
    on_p = on_p / np.sum(on_p, axis=-1, keepdims=True)
    off_p = off_p / np.sum(off_p, axis=-1, keepdims=True)
    eps = 1e-12
    on_p = np.clip(on_p, eps, 1.0)
    off_p = np.clip(off_p, eps, 1.0)
    tv_distance = 0.5 * np.sum(np.abs(on_p - off_p), axis=-1)
    kl_drift = np.sum(on_p * (np.log(on_p) - np.log(off_p)), axis=-1)
    return {
        "tv_distance": float(np.mean(tv_distance)),
        "kl_drift": float(np.mean(kl_drift)),
        "max_tv": float(np.max(tv_distance))
    }
