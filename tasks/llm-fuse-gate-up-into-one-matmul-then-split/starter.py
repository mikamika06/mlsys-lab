import numpy as np

def fused_swiglu(x, w_up, b_up, w_gate, b_gate):
    """Broken implementation – missing the swish activation on the gate.
This will produce incorrect results and fail the max_abs_err metric."""
    raise NotImplementedError('your code here')
