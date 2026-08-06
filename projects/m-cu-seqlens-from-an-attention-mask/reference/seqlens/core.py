import numpy as np


def extract_cu_seqlens(mask):
    lengths = mask.sum(axis=1).astype(np.int32)
    cu_seqlens = np.zeros((len(lengths) + 1,), dtype=np.int32)
    cu_seqlens[1:] = np.cumsum(lengths)
    return cu_seqlens, lengths


def detect_leakage(attn_weights, cu_seqlens):
    batch_size = len(cu_seqlens) - 1
    for i in range(batch_size):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        for j in range(batch_size):
            if i == j:
                continue
            other_start = cu_seqlens[j]
            other_end = cu_seqlens[j+1]
            if other_end <= other_start:
                continue
            sub_block = attn_weights[start:end, other_start:other_end]
            if np.any(np.abs(sub_block) > 1e-7):
                return True
    return False
