import numpy as np


def detect_leakage(attention_weights, cu_seqlens):
    total = cu_seqlens[-1]
    if attention_weights.shape[-2] != total or attention_weights.shape[-1] != total:
        return True
    valid_mask = np.zeros((total, total), dtype=bool)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i+1]
        valid_mask[s:e, s:e] = True
    leaked = np.any((attention_weights > 0) & (~valid_mask))
    return bool(leaked)
