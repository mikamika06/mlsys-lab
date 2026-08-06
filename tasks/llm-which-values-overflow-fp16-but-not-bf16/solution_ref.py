import numpy as np

def which_overflow_fp16_not_bf16(arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating elements that would overflow when cast to fp16
    but not when cast to bf16.
    """
    arr = np.asarray(arr)
    fp16_max = float(np.finfo(np.float16).max)
    try:
        bf16_max = float(np.finfo(np.bfloat16).max)
    except AttributeError:
        bf16_max = (2.0 - 2**-7) * 2**127

    out = np.empty(arr.shape, dtype=bool)
    arr_flat = arr.flat
    out_flat = out.flat

    for i in range(arr.size):
        val = float(arr_flat[i])
        
        if val < 0.0:
            abs_val = -val
        else:
            abs_val = val

        cond1 = abs_val > fp16_max
        cond2 = not (abs_val > bf16_max)
        
        out_flat[i] = cond1 and cond2

    return out
