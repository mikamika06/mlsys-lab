import numpy as np

def compute_offsets(batch_meta):
    cu = batch_meta["cu_seqlens"]
    offsets = []
    for i in range(len(cu) - 1):
        start = cu[i]
        end = cu[i+1]
        offsets.append((start, end))
    return np.array(offsets, dtype=np.int32)
