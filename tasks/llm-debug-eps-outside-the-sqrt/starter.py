import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm — but there is a bug on the std line. Fix it."""
    mu  = x.mean()
    var = ((x - mu) ** 2).mean()
    std = var ** 0.5 + eps        # BUG: eps is outside the sqrt
    return gamma * (x - mu) / std + beta
