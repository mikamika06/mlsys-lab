import numpy as np


def compute_gkd_loss(model_logits, ref_logits, target_ids, beta, theta):
    p = np.exp(ref_logits - np.max(ref_logits, axis=-1, keepdims=True))
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(model_logits - np.max(model_logits, axis=-1, keepdims=True))
    q /= np.sum(q, axis=-1, keepdims=True)
    vocab_size = p.shape[-1]
    target_one_hot = np.eye(vocab_size)[target_ids]
    mixed_p = theta * p + (1 - theta) * target_one_hot
    if beta == 0.0:
        log_q = np.log(np.clip(q, 1e-12, 1.0))
        loss = -np.sum(mixed_p * log_q, axis=-1)
    elif beta == 1.0:
        log_p = np.log(np.clip(mixed_p, 1e-12, 1.0))
        log_q = np.log(np.clip(q, 1e-12, 1.0))
        loss = np.sum(mixed_p * (log_p - log_q), axis=-1)
    else:
        log_p = np.log(np.clip(mixed_p, 1e-12, 1.0))
        log_q = np.log(np.clip(q, 1e-12, 1.0))
        term = (1.0 / (beta * (1.0 - beta))) * (
            np.sum(mixed_p, axis=-1) - np.sum(np.power(mixed_p, 1.0 - beta) * np.power(q, beta), axis=-1)
        )
        loss = term
    return float(np.mean(loss))
