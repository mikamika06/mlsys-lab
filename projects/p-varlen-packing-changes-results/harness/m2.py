import numpy as np
import ref

def check(workdir):
    from pack.attention import align_causal_mask
    m = {"mask_aligned": 0.0}
    cu_seqlens = np.array([0, 4, 10], dtype=np.int32)
    mask = align_causal_mask(cu_seqlens)
    if mask.shape == (10, 10):
        m["mask_aligned"] = 1.0
    return m
