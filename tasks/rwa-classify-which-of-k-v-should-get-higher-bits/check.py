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
    # avoid division by zero when all values equal
    quantized = np.round((arr - v_min) / scale) * scale + v_min
    return quantized

def _attention_output(K, V):
    d = K.shape[1]
    scores = (K @ K.T) / np.sqrt(d)
    w = _softmax(scores)
    return w @ V

def grade(sol, fx) -> dict:
    # generate a few random test cases
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(5):
        n = rng.integers(3, 8)
        d = rng.integers(2, 6)
        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))
        total_bits = 8
        cases.append((K, V, total_bits))

    ok = 1.0
    for K, V, tb in cases:
        try:
            student_idx = sol.classify_high_bits(K.tolist(), V.tolist(), tb)
        except Exception:
            return {"argmin_index": 0.0}

        # reference full‑precision output
        ref_out = _attention_output(K, V)

        # two allocations: K high / V low and vice versa
        bits_high = tb - 1
        bits_low = 1

        K_hi = _quantize(K, bits_high)
        V_lo = _quantize(V, bits_low)
        out_a = _attention_output(K_hi, V_lo)

        K_lo = _quantize(K, bits_low)
        V_hi = _quantize(V, bits_high)
        out_b = _attention_output(K_lo, V_hi)

        mse_a = np.mean((out_a - ref_out)**2)
        mse_b = np.mean((out_b - ref_out)**2)

        oracle_idx = 0 if mse_a <= mse_b else 1

        if student_idx != oracle_idx:
            ok = 0.0
            break

    return {"argmin_index": ok}
