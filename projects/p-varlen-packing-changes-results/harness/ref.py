import numpy as np

def compute_reference(q, k, v, cu_seqlens):
    out = np.zeros_like(q)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        qi = q[start:end]
        ki = k[start:end]
        vi = v[start:end]
        scores = (qi @ ki.T) / np.sqrt(q.shape[-1])
        seq_len = end - start
        mask = np.triu(np.full((seq_len, seq_len), -1e4), k=1)
        scores = scores + mask
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out[start:end] = attn @ vi
    return out
