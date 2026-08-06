import numpy as np


def compute_toy_divergence(p_logits, q_logits, mode="forward_kl", beta=0.0):
    p = np.array(p_logits, dtype=np.float64)
    q = np.array(q_logits, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    if mode == "forward_kl":
        val = np.sum(p * (np.log(p) - np.log(q)))
    elif mode == "reverse_kl":
        val = np.sum(q * (np.log(q) - np.log(p)))
    elif mode == "jsd" or mode == "generalized":
        b = beta if mode == "generalized" else 0.5
        m = (1.0 - b) * p + b * q
        term1 = np.sum(p * (np.log(p) - np.log(m)))
        term2 = np.sum(q * (np.log(q) - np.log(m)))
        val = (1.0 - b) * term1 + b * term2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return float(val)
