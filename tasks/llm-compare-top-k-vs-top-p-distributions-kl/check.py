import numpy as np


def _oracle_top_k_top_p_kl(logits, k, p):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    probs = np.exp(x)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    topk_values = np.zeros_like(probs)
    for r in range(probs.shape[0]):
        idx = np.argsort(probs[r])[::-1][:k]
        topk_values[r, idx] = probs[r, idx]
    topk_values = topk_values / np.sum(topk_values, axis=1, keepdims=True)

    topp_values = np.zeros_like(probs)
    for r in range(probs.shape[0]):
        order = np.argsort(probs[r])[::-1]
        cumulative = 0.0
        chosen = []
        for idx in order:
            chosen.append(idx)
            cumulative += probs[r, idx]
            if cumulative >= p:
                break
        topp_values[r, chosen] = probs[r, chosen]
    topp_values = topp_values / np.sum(topp_values, axis=1, keepdims=True)

    eps = 1e-12
    kl = np.sum(
        topk_values * (np.log(topk_values + eps) - np.log(topp_values + eps)),
        axis=1,
    )
    return float(np.mean(kl))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [3.0, 2.0, 1.0, 0.0],
                [0.5, 0.2, -0.1, -1.0],
            ]),
            2,
            0.8,
        ),
        (
            np.array([
                [8.0, 7.0, 6.0, 5.0, 1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]),
            3,
            0.75,
        ),
        (
            np.array([
                [0.0, 0.0, 0.0, 0.0],
            ]),
            1,
            0.5,
        ),
    ]

    ok = 0.0
    for logits, k, p in cases:
        try:
            got = float(sol.top_k_top_p_kl(logits, k, p))
        except Exception:
            return {"mean_kl": float("inf")}
        ref = _oracle_top_k_top_p_kl(logits, k, p)
        if abs(got - ref) > 1e-12:
            return {"mean_kl": abs(got - ref)}
        ok = abs(got - ref)
    return {"mean_kl": ok}
