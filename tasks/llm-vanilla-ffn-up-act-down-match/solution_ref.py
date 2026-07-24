import numpy as np

def ffn_forward(x, W_up, b_up, W_down, b_down):
    """Vanilla FFN forward: down_proj(gelu(up_proj(x)))."""
    h = W_up @ x + b_up
    a = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
    return W_down @ a + b_down
