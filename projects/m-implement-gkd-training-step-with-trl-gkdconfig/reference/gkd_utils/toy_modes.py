import numpy as np


def evaluate_toy_gkd_modes(logits_p, logits_q, beta, mode):
    p = np.exp(logits_p - np.max(logits_p))
    p /= np.sum(p)
    q = np.exp(logits_q - np.max(logits_q))
    q /= np.sum(q)
    if mode == "forward_kl":
        loss = np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0))))
    elif mode == "reverse_kl":
        loss = np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))))
    elif mode == "jsd":
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
        kl_qm = np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
        loss = 0.5 * (kl_pm + kl_qm)
    else:
        if beta == 0.0:
            loss = -np.sum(p * np.log(np.clip(q, 1e-12, 1.0)))
        elif beta == 1.0:
            loss = np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0))))
        else:
            loss = (1.0 / (beta * (1.0 - beta))) * (1.0 - np.sum(np.power(p, 1.0 - beta) * np.power(q, beta)))
    return float(loss)
