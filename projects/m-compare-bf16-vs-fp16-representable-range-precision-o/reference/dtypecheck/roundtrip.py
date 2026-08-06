import numpy as np

def quantify_roundtrip_loss(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    u = arr.view(np.uint32)
    bf16_u = u & 0xFFFF0000
    roundtrip = bf16_u.view(np.float32)
    diff = np.abs(arr - roundtrip)
    rel_err = np.max(diff / (np.abs(arr) + 1e-12))
    return float(rel_err)
