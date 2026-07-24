import numpy as np


def h2o_eviction_set(attn_scores: np.ndarray, budget: int, recent_window: int):
    """
    H2O (Heavy-Hitter Oracle) static eviction set.

    attn_scores : (n, n) raw attention logits.
    budget      : number of tokens to keep (recent_window <= budget <= n).
    recent_window : number of most-recent positions always kept.

    Returns (retained_idx, preserved_mass):
      retained_idx   : 1-D int64 array, ascending, length == budget.
      preserved_mass : float, fraction of total accumulated attention mass
                        captured by the retained set.
    """
    S = np.asarray(attn_scores, dtype=np.float64)
    n = S.shape[0]

    # Causal mask: token i may only attend to j <= i.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    S_masked = np.where(mask, -np.inf, S)
    S_masked = S_masked - np.max(S_masked, axis=1, keepdims=True)
    P = np.exp(S_masked)
    P = P / np.sum(P, axis=1, keepdims=True)

    # Accumulated importance per token (column sums).
    h = P.sum(axis=0)

    recent = set(range(max(0, n - recent_window), n))
    n_heavy = budget - len(recent)

    candidates = [j for j in range(n) if j not in recent]
    # Sort candidates by descending score, tie-break by smaller index.
    candidates.sort(key=lambda j: (-h[j], j))
    heavy = candidates[:n_heavy]

    retained = sorted(set(heavy) | recent)
    retained_idx = np.array(retained, dtype=np.int64)

    preserved_mass = float(h[retained_idx].sum() / h.sum())
    return retained_idx, preserved_mass
