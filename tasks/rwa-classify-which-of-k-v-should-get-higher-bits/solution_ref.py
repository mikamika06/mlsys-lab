import math
import numpy as np

def _softmax(x):
    if x.ndim == 1:
        n = x.shape[0]
        max_val = x[0]
        for i in range(1, n):
            if x[i] > max_val:
                max_val = x[i]
        out = np.empty_like(x)
        s = 0.0
        for i in range(n):
            val = math.exp(x[i] - max_val)
            out[i] = val
            s += val
        for i in range(n):
            out[i] /= s
        return out
    else:
        n_rows, n_cols = x.shape
        out = np.empty_like(x)
        for i in range(n_rows):
            max_val = x[i, 0]
            for j in range(1, n_cols):
                if x[i, j] > max_val:
                    max_val = x[i, j]
            s = 0.0
            for j in range(n_cols):
                val = math.exp(x[i, j] - max_val)
                out[i, j] = val
                s += val
            for j in range(n_cols):
                out[i, j] /= s
        return out

def _quantize(arr, bits):
    if bits <= 0:
        return arr
    qmin = 0.0
    qmax = float(2**bits - 1)
    first = True
    v_min = 0.0
    v_max = 0.0
    for val in arr.flat:
        if first:
            v_min = val
            v_max = val
            first = False
        else:
            if val < v_min:
                v_min = val
            if val > v_max:
                v_max = val
    scale = (v_max - v_min) / (qmax - qmin) if qmax != qmin else 1.0
    quantized = np.empty_like(arr)
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = arr[idx]
        quantized[idx] = round((val - v_min) / scale) * scale + v_min
        it.iternext()
    return quantized

def _attention_output(K, V):
    d = K.shape[1]
    n = K.shape[0]
    scores = np.empty((n, n), dtype=K.dtype)
    sqrt_d = math.sqrt(d)
    for i in range(n):
        for j in range(n):
            dot_val = 0.0
            for k_idx in range(d):
                dot_val += K[i, k_idx] * K[j, k_idx]
            scores[i, j] = dot_val / sqrt_d
    w = _softmax(scores)
    v_cols = V.shape[1]
    out = np.empty((n, v_cols), dtype=w.dtype)
    for i in range(n):
        for j in range(v_cols):
            val = 0.0
            for k_idx in range(n):
                val += w[i, k_idx] * V[k_idx, j]
            out[i, j] = val
    return out

def classify_high_bits(K: np.ndarray, V: np.ndarray, total_bits:int=8) -> int:
    """
    Return 0 if allocating higher precision to K yields lower MSE in the
    attention output than allocating it to V; otherwise return 1.
    """
    ref_out = _attention_output(K, V)

    bits_high = total_bits - 1
    bits_low = 1

    K_hi = _quantize(K, bits_high)
    V_lo = _quantize(V, bits_low)
    out_a = _attention_output(K_hi, V_lo)
    
    sum_sq_a = 0.0
    count_a = 0
    for val_a, val_ref in zip(out_a.flat, ref_out.flat):
        diff = val_a - val_ref
        sum_sq_a += diff * diff
        count_a += 1
    mse_a = sum_sq_a / count_a

    K_lo = _quantize(K, bits_low)
    V_hi = _quantize(V, bits_high)
    out_b = _attention_output(K_lo, V_hi)
    
    sum_sq_b = 0.0
    count_b = 0
    for val_b, val_ref in zip(out_b.flat, ref_out.flat):
        diff = val_b - val_ref
        sum_sq_b += diff * diff
        count_b += 1
    mse_b = sum_sq_b / count_b

    return 0 if mse_a <= mse_b else 1
