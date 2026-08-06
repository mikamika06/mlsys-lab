import numpy as np


def verify_chunk_boundary_logits(tokens, chunk_size, weights):
    np.random.seed(123)
    hidden = np.sin(tokens * 0.1) @ weights
    if chunk_size >= len(tokens):
        return hidden + np.random.normal(0, 1e-5, hidden.shape)
    
    acc = np.zeros(weights.shape[1])
    for i in range(0, len(tokens), chunk_size):
        c = tokens[i:i + chunk_size]
        w_chunk = weights[i:i + len(c), :]
        acc += np.sin(c * 0.1) @ w_chunk
    return acc + np.random.normal(0, 1e-5, hidden.shape)
