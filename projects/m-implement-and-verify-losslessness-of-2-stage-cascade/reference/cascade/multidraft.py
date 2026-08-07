import numpy as np


def sample_optimal_k_drafts(draft_probs, target_probs, seed=42):
    rng = np.random.default_rng(seed)
    K, V = draft_probs.shape

    selected_k = rng.integers(0, K)
    x = rng.choice(V, p=draft_probs[selected_k])

    q_max = np.max(draft_probs[:, x])
    p_x = target_probs[x]

    u = rng.uniform()
    accept_prob = min(1.0, p_x / (K * q_max))

    if u < accept_prob:
        return int(x), True

    avg_q = np.mean(draft_probs, axis=0)
    resample_p = np.maximum(0.0, target_probs - avg_q)
    s = resample_p.sum()
    if s > 0:
        resample_p = resample_p / s
    else:
        resample_p = target_probs.copy()

    resample_tok = rng.choice(V, p=resample_p)
    return int(resample_tok), False
