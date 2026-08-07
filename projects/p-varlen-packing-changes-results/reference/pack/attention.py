import numpy as np

def detect_boundary(q, cu_seqlens):
    ranges = []
    for i in range(len(cu_seqlens) - 1):
        ranges.append((cu_seqlens[i], cu_seqlens[i+1]))
    return ranges

def align_causal_mask(cu_seqlens):
    total = cu_seqlens[-1]
    mask = np.full((total, total), -1e4)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        seq_len = end - start
        sub_mask = np.triu(np.full((seq_len, seq_len), -1e4), k=1)
        mask[start:end, start:end] = sub_mask
    return mask

def process_cu_seqlens(cu_seqlens):
    offsets = []
    for i in range(len(cu_seqlens) - 1):
        offsets.append((cu_seqlens[i], cu_seqlens[i+1]))
    return offsets

def varlen_attention(q, k, v, cu_seqlens):
    total = cu_seqlens[-1]
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
