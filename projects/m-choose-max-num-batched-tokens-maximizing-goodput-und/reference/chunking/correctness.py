import numpy as np


def verify_chunked_prefill_logits(full_logits, chunked_logits, tol=1e-5):
    if len(full_logits) != len(chunked_logits):
        return False
    for f, c in zip(full_logits, chunked_logits):
        f_arr = np.array(f, dtype=float)
        c_arr = np.array(c, dtype=float)
        if f_arr.shape != c_arr.shape:
            return False
        if not np.allclose(f_arr, c_arr, atol=tol, rtol=tol):
            return False
    return True
