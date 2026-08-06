import math
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shape = logits.shape
    last_dim = shape[-1]
    num_slices = 1
    for d in shape[:-1]:
        num_slices *= d
    flat_input = logits.reshape((num_slices, last_dim))
    out = np.empty_like(flat_input, dtype=np.float64)
    for i in range(num_slices):
        row = flat_input[i]
        m = float(row[0])
        for j in range(1, last_dim):
            v = float(row[j])
            if v > m:
                m = v
        exps = [0.0] * last_dim
        s = 0.0
        for j in range(last_dim):
            e = math.exp(float(row[j]) - m)
            exps[j] = e
            s += e
        for j in range(last_dim):
            out[i, j] = exps[j] / s
    return out.reshape(shape)
