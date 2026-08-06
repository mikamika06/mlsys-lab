import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm with eps correctly placed INSIDE the sqrt."""
    n = len(x)
    
    total = 0.0
    for i in range(n):
        total += x[i]
    mu = total / n

    var_total = 0.0
    for i in range(n):
        diff = x[i] - mu
        var_total += diff * diff
    var = var_total / n

    std = (var + eps) ** 0.5

    out = np.empty(n, dtype=x.dtype)
    for i in range(n):
        out[i] = gamma[i] * (x[i] - mu) / std + beta[i]

    return out
