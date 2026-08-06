import math
import numpy as np

def ffn_forward(x, W_up, b_up, W_down, b_down):
    """Vanilla FFN forward: down_proj(gelu(up_proj(x)))."""
    d_hidden = W_up.shape[0]
    d = W_down.shape[0]
    
    h = np.zeros(d_hidden, dtype=W_up.dtype)
    for i in range(d_hidden):
        acc = b_up[i]
        for j in range(x.shape[0]):
            acc += W_up[i, j] * x[j]
        h[i] = acc
        
    a = np.zeros(d_hidden, dtype=h.dtype)
    const = math.sqrt(2.0 / math.pi)
    for i in range(d_hidden):
        val = h[i]
        gelu_val = 0.5 * val * (1.0 + math.tanh(const * (val + 0.044715 * (val ** 3))))
        a[i] = gelu_val
        
    res = np.zeros(d, dtype=W_down.dtype)
    for i in range(d):
        acc = b_down[i]
        for j in range(d_hidden):
            acc += W_down[i, j] * a[j]
        res[i] = acc
        
    return res
