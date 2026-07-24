import numpy as np

def mha_single_kv_head(Q, K, V):
    """Compute scaled dot-product attention with a single shared KV head.

    Q : (B, H, S, D)  – queries, H heads
    K : (B, 1, S, D)  – keys, one head
    V : (B, 1, S, D)  – values, one head

    Returns (B, H, S, D) via NumPy broadcasting (no explicit KV expansion).
    """
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5

    # Q @ K^T  with broadcasting:
    #   Q      : (B, H, S, D)
    #   K^T    : (B, 1, D, S)  ->  scores: (B, H, S, S)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale

    # Numerically stable softmax over the last axis
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # weights @ V  with broadcasting:
    #   weights: (B, H, S, S)
    #   V      : (B, 1, S, D)  ->  output: (B, H, S, D)
    output = np.matmul(weights, V)

    return output
