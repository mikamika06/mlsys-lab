import numpy as np

def business_logic_model(x, seq_len):
    out = np.zeros_like(x)
    for i in range(seq_len):
        val = x[i] if i < len(x) else 0.0
        if val > 0.5:
            out[i] = val * 2.0
        else:
            out[i] = val + 1.0
    return out
