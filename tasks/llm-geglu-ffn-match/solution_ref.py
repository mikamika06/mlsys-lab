import math
import numpy as np


def geglu_ffn(x, w_gate, w_up):
    """Gated Linear Unit with GELU activation."""
    x = np.asarray(x, dtype=np.float64)
    w_gate = np.asarray(w_gate, dtype=np.float64)
    w_up = np.asarray(w_up, dtype=np.float64)

    orig_shape = x.shape
    d_in = w_gate.shape[0]
    d_out = w_gate.shape[1]

    out_shape = orig_shape[:-1] + (d_out,)
    x_2d = x.reshape(-1, d_in)
    rows = x_2d.shape[0]

    out_2d = np.zeros((rows, d_out), dtype=np.float64)

    sqrt_2_pi = math.sqrt(2.0 / math.pi)

    for i in range(rows):
        for j in range(d_out):
            gate_val = 0.0
            up_val = 0.0
            for k in range(d_in):
                gate_val += float(x_2d[i, k]) * float(w_gate[k, j])
                up_val += float(x_2d[i, k]) * float(w_up[k, j])

            val = sqrt_2_pi * (gate_val + 0.044715 * gate_val * gate_val * gate_val)
            gelu = 0.5 * gate_val * (1.0 + math.tanh(val))
            out_2d[i, j] = gelu * up_val

    return out_2d.reshape(out_shape)
