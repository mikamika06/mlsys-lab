import ref
import numpy as np


def check(workdir):
    from q4k.quantize import quantize_q4_k, dequantize_q4_k
    t = ref.get_test_tensor()
    try:
        b = quantize_q4_k(t)
        b_ref = ref.quantize_q4_k(t)
        match = 1.0 if b == b_ref else 0.0
        return {"bytes_matched": match}
    except Exception as e:
        return {"bytes_matched": 0.0, "_note": str(e)}
