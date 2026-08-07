import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    from quant.recipes import quantize_int4, quantize_int8

    out = {
        "int8_compression_matches": 0.0,
        "int4_compression_matches": 0.0,
        "recipe_eval_correct": 0.0,
    }

    w = np.linspace(-10, 10, 100, dtype=np.float32)

    try:
        q8, scale8, deq8 = quantize_int8(w)
        q4, scale4, deq4 = quantize_int4(w)
    except Exception:
        return out

    if q8.dtype == np.int8:
        out["int8_compression_matches"] = 1.0
    if q4.dtype == np.int8:
        out["int4_compression_matches"] = 1.0

    err8 = np.mean(np.abs(w - deq8))
    err4 = np.mean(np.abs(w - deq4))

    if err8 <= err4 and np.max(np.abs(q4)) <= 8:
        out["recipe_eval_correct"] = 1.0

    return out
