import numpy as np

def convert_outtype(weights, outtype):
    out = {}
    for k, v in weights.items():
        if outtype == "f16":
            out[k] = v.astype(np.float16)
        elif outtype == "bf16":
            arr = v.astype(np.float32)
            u32 = arr.view(np.uint32)
            u16 = (u32 >> 16).astype(np.uint16)
            out[k] = u16.view(np.bfloat16) if hasattr(np, "bfloat16") else u16
        else:
            out[k] = v.astype(np.float32)
    return out
