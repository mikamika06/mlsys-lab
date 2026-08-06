import numpy as np

def cast_outtype(tensor, outtype):
    arr = np.asarray(tensor, dtype=np.float32)
    if outtype == "f16":
        return arr.astype(np.float16).astype(np.float32)
    elif outtype == "bf16":
        u32 = arr.view(np.uint32)
        u32_bf = u32 & 0xFFFF0000
        return u32_bf.view(np.float32)
    return arr
