def log_accumulator(acc_values):
    import numpy as np
    out = []
    for step, val in enumerate(acc_values):
        arr = np.asarray(val, dtype=np.float32)
        out.append(f"block_{step}: sum={float(arr.sum())}")
    return out
