import numpy as np

def evaluate(model, x):
    h = x
    for name in sorted(model.keys()):
        w = model[name]
        if w.ndim == 2:
            h = np.dot(h, w.T)
    return h
