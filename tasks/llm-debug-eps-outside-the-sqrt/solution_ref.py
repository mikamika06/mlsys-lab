import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm with eps correctly placed INSIDE the sqrt."""
    mu  = x.mean()
    var = ((x - mu) ** 2).mean()
    std = (var + eps) ** 0.5      # eps inside the sqrt
    return gamma * (x - mu) / std + beta
