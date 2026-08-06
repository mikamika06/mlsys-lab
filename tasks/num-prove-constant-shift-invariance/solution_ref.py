import numpy as np
import math

def _stable_softmax(x):
    """Numerically stable softmax: subtract max, exponentiate, normalize."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = x.shape[0]
        max_val = x[0]
        for i in range(1, n):
            if x[i] > max_val:
                max_val = x[i]
        
        out = np.empty_like(x)
        sum_e = 0.0
        for i in range(n):
            val = math.exp(x[i] - max_val)
            out[i] = val
            sum_e += val
        
        for i in range(n):
            out[i] /= sum_e
        return out
    elif x.ndim == 2:
        rows, cols = x.shape
        out = np.empty_like(x)
        for i in range(rows):
            max_val = x[i, 0]
            for j in range(1, cols):
                if x[i, j] > max_val:
                    max_val = x[i, j]
            
            sum_e = 0.0
            for j in range(cols):
                val = math.exp(x[i, j] - max_val)
                out[i, j] = val
                sum_e += val
            
            for j in range(cols):
                out[i, j] /= sum_e
        return out
    else:
        raise ValueError("Only 1D and 2D arrays are supported")

def softmax_shift_invariant(logits, shift):
    """
    Returns the maximum absolute error between softmax(logits) and
    softmax(logits - shift), proving numerical invariance to constant shifts.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)

    soft_original = _stable_softmax(logits)
    
    if logits.ndim == 1:
        n = logits.shape[0]
        shifted_logits = np.empty_like(logits)
        for i in range(n):
            shifted_logits[i] = logits[i] - shift[i]
    elif logits.ndim == 2:
        rows, cols = logits.shape
        shifted_logits = np.empty_like(logits)
        for i in range(rows):
            for j in range(cols):
                shifted_logits[i, j] = logits[i, j] - shift[i, j]
    else:
        shifted_logits = logits - shift

    soft_shifted = _stable_softmax(shifted_logits)

    max_err = 0.0
    if soft_original.ndim == 1:
        n = soft_original.shape[0]
        for i in range(n):
            val = soft_original[i] - soft_shifted[i]
            abs_val = -val if val < 0 else val
            if abs_val > max_err:
                max_err = abs_val
    elif soft_original.ndim == 2:
        rows, cols = soft_original.shape
        for i in range(rows):
            for j in range(cols):
                val = soft_original[i, j] - soft_shifted[i, j]
                abs_val = -val if val < 0 else val
                if abs_val > max_err:
                    max_err = abs_val

    return float(max_err)
