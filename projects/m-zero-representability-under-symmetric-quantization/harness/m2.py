import ref
import numpy as np


def check(workdir):
    from quant.sym import quantize, dequantize, compute_scale
    out = {"zero_exact_match": 0.0}
    for t in ref.TENSORS:
        scale = compute_scale(t, -128, 127)
        codes = quantize(t, scale, -128, 127)
        deq = dequantize(codes, scale)
        zero_mask = (t == 0.0)
        if np.any(zero_mask):
            if not np.all(deq[zero_mask] == 0.0):
                return out
    out["zero_exact_match"] = 1.0
    return out
