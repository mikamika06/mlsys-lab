import numpy as np

def geglu_ffn(x, w_gate, w_up):
    x = np.asarray(x, dtype=np.float64)
    w_gate = np.asarray(w_gate, dtype=np.float64)
    w_up   = np.asarray(w_up, dtype=np.float64)

    gate = x @ w_gate
    up   = x @ w_up

    gelu = 0.5 * gate * (1 + np.tanh(np.sqrt(2/np.pi) * (gate + 0.044715 * gate**3)))
    return gelu * up
