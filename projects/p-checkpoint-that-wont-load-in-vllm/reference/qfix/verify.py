import numpy as np

def verify_output(weights, inputs):
    res = 0.0
    for k, w in weights.items():
        if w.size > 0:
            res += float(np.sum(w) * np.sum(inputs))
    return res
