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

    order = np.argsort(-logits, axis=1, kind="stable")
    indices = order[:, :k]

    rows = np.arange(N)[:, None]
    top_logits = logits[rows, indices]
    m = np.max(top_logits, axis=1, keepdims=True)
    e = np.exp(top_logits - m)
    weights = e / np.sum(e, axis=1, keepdims=True)

    return indices.astype(np.int64), weights
