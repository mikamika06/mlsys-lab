import numpy as np


def top_k_top_p_kl(logits, k, p):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    probs = np.exp(x)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    topk = np.zeros_like(probs)
    topp = np.zeros_like(probs)

    for r in range(probs.shape[0]):
        idx = np.argsort(probs[r])[::-1][:k]
        topk[r, idx] = probs[r, idx]

        order = np.argsort(probs[r])[::-1]
        total = 0.0
        for idx in order:
            topp[r, idx] = probs[r, idx]
            total += probs[r, idx]
            if total >= p:
                break

    topk = topk / np.sum(topk, axis=1, keepdims=True)
    topp = topp / np.sum(topp, axis=1, keepdims=True)

    eps = 1e-12
    return float(
        np.mean(
            np.sum(
                topk * (np.log(topk + eps) - np.log(topp + eps)),
                axis=1,
            )
        )
    )
