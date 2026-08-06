import numpy as np


def kl_divergence_to_acceptance_bound(p_draft, p_target):
    """Compute KL divergence and lower bound on expected acceptance probability."""
    p_draft = np.asarray(p_draft, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    p_draft = np.clip(p_draft, 1e-12, 1.0)
    p_draft = p_draft / np.sum(p_draft)

    p_target = np.clip(p_target, 1e-12, 1.0)
    p_target = p_target / np.sum(p_target)

    kl_div = np.sum(p_draft * np.log(p_draft / p_target))
    tv_dist = 0.5 * np.sum(np.abs(p_draft - p_target))
    lower_bound = max(0.0, float(1.0 - tv_dist))

    return float(kl_div), lower_bound
