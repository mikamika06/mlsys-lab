import numpy as np
from adaptermerge.merge import merge_adapters
from adaptermerge.drift import compute_relative_error

def evaluate_drift(w_base, delta1, delta2, x, scale1=1.0, scale2=1.0):
    out_ref = np.dot(x, w_base) + scale1 * np.dot(x, delta1) + scale2 * np.dot(x, delta2)
    w_merged = merge_adapters(w_base, delta1, delta2, scale1, scale2)
    out_merged = np.dot(x, w_merged)
    return compute_relative_error(out_ref, out_merged)
