import numpy as np


def evaluate_mode_behavior(p_probs, q_probs, mode):
    p = np.clip(np.asarray(p_probs, dtype=np.float32), 1e-10, 1.0)
    q = np.clip(np.asarray(q_probs, dtype=np.float32), 1e-10, 1.0)

    p = p / np.sum(p)
    q = q / np.sum(q)

    if mode == "forward_kl":
        div = np.sum(p * (np.log(p) - np.log(q)))
    elif mode == "reverse_kl":
        div = np.sum(q * (np.log(q) - np.log(p)))
    elif mode == "jsd":
        m = 0.5 * (p + q)
        div = 0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(q * (np.log(q) - np.log(m)))
    else:
        div = np.sum(p * (np.log(p) - np.log(q)))

    entropy = -np.sum(p * np.log(p))
    return {"divergence": float(div), "entropy": float(entropy)}
