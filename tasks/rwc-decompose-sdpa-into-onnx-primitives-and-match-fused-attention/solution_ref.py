import numpy as np

def decompose_sdpa(Q, K, V, mask=None, scale=None):
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # Step 1: MatMul — raw attention scores  S = Q @ Kᵀ
    scores = Q @ np.swapaxes(K, -2, -1)

    # Step 2: Mul — scale the logits
    scores = scores * scale

    # Step 3: Add — apply additive mask (skip when absent)
    if mask is not None:
        scores = scores + mask

    # Step 4: Softmax — numerically stable, axis = -1
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: MatMul — output = W @ V
    output = weights @ V

    return output
