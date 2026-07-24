import numpy as np

def which_overflow_fp16_not_bf16(arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating elements that would overflow when cast to fp16
    but not when cast to bf16.
    """
    arr = np.asarray(arr)
    fp16_max = np.finfo(np.float16).max
    try:
        bf16_max = np.finfo(np.bfloat16).max
    except AttributeError:
        # Manual computation: (2 - 2^-7) * 2^127
        bf16_max = (2.0 - 2**-7) * 2**127
    return (np.abs(arr) > fp16_max) & (~(np.abs(arr) > bf16_max))
