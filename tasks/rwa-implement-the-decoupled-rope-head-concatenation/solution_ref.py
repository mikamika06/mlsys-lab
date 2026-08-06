import math
import numpy as np

def decoupled_rope_score(q_lat, k_lat, q_rope, k_rope):
    """Concatenate latent and rope-head, then scaled dot-product + softmax."""
    B, H, N, D_l = q_lat.shape
    D_r = q_rope.shape[-1]
    D = D_l + D_r
    
    Q = np.concatenate([q_lat, q_rope], axis=-1)
    K = np.concatenate([k_lat, k_rope], axis=-1)
    
    scale = 1.0 / math.sqrt(D)
    
    out = np.empty((B, H, N, N), dtype=Q.dtype)
    
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(N):
                    dot = 0.0
                    for d in range(D):
                        dot += Q[b, h, i, d] * K[b, h, j, d]
                    out[b, h, i, j] = dot * scale
                    
    for b in range(B):
        for h in range(H):
            for i in range(N):
                max_val = out[b, h, i, 0]
                for j in range(1, N):
                    if out[b, h, i, j] > max_val:
                        max_val = out[b, h, i, j]
                
                sum_exp = 0.0
                for j in range(N):
                    val = math.exp(out[b, h, i, j] - max_val)
                    out[b, h, i, j] = val
                    sum_exp += val
                    
                for j in range(N):
                    out[b, h, i, j] /= sum_exp
                    
    return out
