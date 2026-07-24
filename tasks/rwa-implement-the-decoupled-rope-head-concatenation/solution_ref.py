import numpy as np

def decoupled_rope_score(q_lat, k_lat, q_rope, k_rope):
    """Concatenate latent and rope-head, then scaled dot-product + softmax."""
    D = q_lat.shape[-1] + q_rope.shape[-1]
    Q = np.concatenate([q_lat, q_rope], axis=-1)
    K = np.concatenate([k_lat, k_rope], axis=-1)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)
