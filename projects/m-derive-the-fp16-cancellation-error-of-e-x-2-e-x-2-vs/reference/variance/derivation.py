import numpy as np

def compute_fp16_variance(x):
    x16 = x.astype(np.float16).astype(np.float32)
    m2 = np.mean(x16 ** 2, axis=-1)
    sm = np.mean(x16, axis=-1) ** 2
    sp = m2 - sm
    tp = np.mean((x16 - np.mean(x16, axis=-1, keepdims=True)) ** 2, axis=-1)
    err = np.mean(np.abs(sp - tp) / (np.abs(tp) + 1e-5))
    return float(err)
