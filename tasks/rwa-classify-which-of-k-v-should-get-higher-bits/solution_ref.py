import numpy as np

def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def _quantize(arr, bits):
    if bits <= 0:
        return arr
    qmin = 0.0
    qmax = float(2**bits - 1)
    v_min = arr.min()
    v_max = arr.max()
    scale = (v_max - v_min) / (qmax - qmin) if qmax != qmin else 1.0
    quantized = np.round((arr - v_min) / scale) * scale + v_min
    return quantized

def _attention_output(K, V):
    d = K.shape[1]
    scores = (K @ K.T) / np.sqrt(d)
    w = _softmax(scores)
    return w @ V

def classify_high_bits(K: np.ndarray, V: np.ndarray, total_bits:int=8) -> int:
    """
    Return 0 if allocating higher precision to K yields lower MSE in the
    attention output than allocating it to V; otherwise return 1.
    """
    ref_out = _attention_output(K, V)

    bits_high = total_bits - 1
    bits_low = 1

    # K high / V low
    K_hi = _quantize(K, bits_high)
    V_lo = _quantize(V, bits_low)
    out_a = _attention_output(K_hi, V_lo)
    mse_a = np.mean((out_a - ref_out)**2)

    # K low / V high
    K_lo = _quantize(K, bits_low)
    V_hi = _quantize(V, bits_high)
    out_b = _attention_output(K_lo, V_hi)
    mse_b = np.mean((out_b - ref_out)**2)

    return 0 if mse_a <= mse_b else 1
