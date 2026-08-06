import numpy as np


def compute_output_error(base_w, adapter_a, adapter_b, scaling, x):
    unfused = np.matmul(x, base_w) + scaling * np.matmul(np.matmul(x, adapter_a), adapter_b)
    fused_w = base_w + scaling * np.matmul(adapter_a, adapter_b)
    fused = np.matmul(x, fused_w)
    return float(np.max(np.abs(fused - unfused)))
