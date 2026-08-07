import numpy as np

def fix_packing(tensors):
    fixed = {}
    for k, v in tensors.items():
        if "weight" in k and v.dtype == np.uint8:
            unpacked = np.zeros(v.shape[0] * 2, dtype=np.int8)
            unpacked[0::2] = (v & 0x0F).astype(np.int8)
            unpacked[1::2] = ((v >> 4) & 0x0F).astype(np.int8)
            fixed[k] = unpacked
        else:
            fixed[k] = v
    return fixed
