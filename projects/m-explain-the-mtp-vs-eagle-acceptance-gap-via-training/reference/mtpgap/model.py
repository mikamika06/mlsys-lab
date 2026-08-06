import numpy as np


def compute_mtp_loss(logits, targets, weights):
    losses = []
    for i, l in enumerate(logits):
        t = targets[i]
        exp_l = np.exp(l - np.max(l, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        n = t.shape[0]
        ce = -np.sum(np.log(probs[np.arange(n), t] + 1e-12)) / n
        losses.append(float(ce * weights[i]))
    return float(np.sum(losses))


def compute_eagle_loss(logits, targets):
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    n = targets.shape[0]
    ce = -np.sum(np.log(probs[np.arange(n), targets] + 1e-12)) / n
    return float(ce)


def estimate_gradient_interference(grads_list):
    g1, g2 = grads_list[0], grads_list[1]
    dot = np.sum(g1 * g2)
    norm1 = np.linalg.norm(g1)
    norm2 = np.linalg.norm(g2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))
