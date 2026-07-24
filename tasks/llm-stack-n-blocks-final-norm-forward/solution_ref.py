import numpy as np

def stack_blocks_forward(x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f,
                         n_blocks):
    """Apply N identical residual-MLP blocks then a final LayerNorm."""
    eps = 1e-5
    h = x.copy()
    for _ in range(n_blocks):
        # Block LayerNorm
        mu = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True, ddof=0)
        ln = gamma * ((h - mu) / np.sqrt(var + eps)) + beta
        # MLP sub-layer
        mid = np.maximum(0.0, ln @ W1 + b1)
        out = mid @ W2 + b2
        # Residual connection
        h = h + out
    # Final LayerNorm
    mu_f = h.mean(axis=-1, keepdims=True)
    var_f = h.var(axis=-1, keepdims=True, ddof=0)
    h = gamma_f * ((h - mu_f) / np.sqrt(var_f + eps)) + beta_f
    return h
