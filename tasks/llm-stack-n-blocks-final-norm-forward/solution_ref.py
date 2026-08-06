import numpy as np
import math

def stack_blocks_forward(x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f,
                         n_blocks):
    """Apply N identical residual-MLP blocks then a final LayerNorm."""
    eps = 1e-5
    batch = x.shape[0]
    d = x.shape[1]
    d_hidden = W1.shape[1]
    h = [[x[i, j] for j in range(d)] for i in range(batch)]
    for _ in range(n_blocks):
        ln = [[0.0] * d for _ in range(batch)]
        for i in range(batch):
            mu_val = 0.0
            for j in range(d):
                mu_val += h[i][j]
            mu_val /= d
            var_val = 0.0
            for j in range(d):
                diff = h[i][j] - mu_val
                var_val += diff * diff
            var_val /= d
            inv_std = 1.0 / math.sqrt(var_val + eps)
            for j in range(d):
                ln[i][j] = gamma[j] * ((h[i][j] - mu_val) * inv_std) + beta[j]
        mid = [[0.0] * d_hidden for _ in range(batch)]
        for i in range(batch):
            for k in range(d_hidden):
                dot = 0.0
                for j in range(d):
                    dot += ln[i][j] * W1[j, k]
                val = dot + b1[k]
                mid[i][k] = val if val > 0.0 else 0.0
        out = [[0.0] * d for _ in range(batch)]
        for i in range(batch):
            for j in range(d):
                dot = 0.0
                for k in range(d_hidden):
                    dot += mid[i][k] * W2[k, j]
                out[i][j] = dot + b2[j]
        for i in range(batch):
            for j in range(d):
                h[i][j] += out[i][j]
    h_final = [[0.0] * d for _ in range(batch)]
    for i in range(batch):
        mu_val = 0.0
        for j in range(d):
            mu_val += h[i][j]
        mu_val /= d
        var_val = 0.0
        for j in range(d):
            diff = h[i][j] - mu_val
            var_val += diff * diff
        var_val /= d
        inv_std = 1.0 / math.sqrt(var_val + eps)
        for j in range(d):
            h_final[i][j] = gamma_f[j] * ((h[i][j] - mu_val) * inv_std) + beta_f[j]
    return np.array(h_final, dtype=x.dtype)
