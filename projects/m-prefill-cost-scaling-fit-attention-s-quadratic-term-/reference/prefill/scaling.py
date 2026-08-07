import numpy as np

def fit_scaling(lengths, times):
    p = np.polyfit(lengths, times, 2)
    return {"linear": float(p[1]), "quadratic": float(p[0])}
