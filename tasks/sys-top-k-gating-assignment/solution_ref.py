import math
import numpy as np


def topk_gating(logits: np.ndarray, k: int):
    """Top-k MoE router gating.

    logits : (N, E) float64 router logits, one row per token.
    k : number of experts to route each token to.

    For each token i, select the k experts with the highest logits[i],
    ordered by decreasing logit value (ties broken by the lower expert
    index first). The gate weight for each selected expert is the softmax
    computed ONLY over that token's k selected logits (so the k weights
    for a token sum to 1 -- unselected experts get no weight at all,
    matching standard sparse MoE routing).

    Returns
    -------
    indices : (N, k) int64 -- selected expert index per token, per rank.
    weights : (N, k) float64 -- softmax gate weight per selected expert.
    """
    logits = np.asarray(logits, dtype=np.float64)
    N, E = logits.shape

    indices_rows = []
    weights_rows = []

    for i in range(N):
        row = logits[i]
        sorted_items = sorted(enumerate(row), key=lambda x: (-x[1], x[0]))

        top_indices = [item[0] for item in sorted_items[:k]]
        top_vals = [item[1] for item in sorted_items[:k]]

        m = max(top_vals)
        e_vals = [math.exp(val - m) for val in top_vals]
        s = sum(e_vals)
        w_vals = [ev / s for ev in e_vals]

        indices_rows.append(top_indices)
        weights_rows.append(w_vals)

    indices = np.array(indices_rows, dtype=np.int64)
    weights = np.array(weights_rows, dtype=np.float64)

    return indices, weights
