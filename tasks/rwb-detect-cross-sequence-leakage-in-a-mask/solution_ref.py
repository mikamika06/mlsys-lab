import numpy as np

def detect_leakage(mask: np.ndarray, cu_seqlens: np.ndarray) -> bool:
    N = mask.shape[0]
    seg_ids = np.empty(N, dtype=np.int32)
    for i in range(len(cu_seqlens)-1):
        seg_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i
    return bool(np.any((seg_ids[:,None] != seg_ids[None,:]) & mask.astype(bool)))
