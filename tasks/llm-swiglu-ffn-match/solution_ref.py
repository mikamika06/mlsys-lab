import numpy as np


def swiglu_ffn(x, gate_w, up_w, down_w):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ np.asarray(gate_w, dtype=np.float64)
    up = x @ np.asarray(up_w, dtype=np.float64)
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    return hidden @ np.asarray(down_w, dtype=np.float64)
