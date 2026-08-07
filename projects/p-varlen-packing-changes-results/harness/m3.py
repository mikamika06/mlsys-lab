import numpy as np
import ref

def check(workdir):
    from pack.attention import process_cu_seqlens
    m = {"cu_seqlens_ok": 0.0}
    cu_seqlens = np.array([0, 3, 8], dtype=np.int32)
    offsets = process_cu_seqlens(cu_seqlens)
    if len(offsets) == 2:
        m["cu_seqlens_ok"] = 1.0
    return m
