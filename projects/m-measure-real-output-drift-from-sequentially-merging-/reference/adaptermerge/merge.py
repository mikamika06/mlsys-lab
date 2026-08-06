import numpy as np

def merge_adapters(w_base, delta1, delta2, scale1=1.0, scale2=1.0):
    return w_base + scale1 * delta1 + scale2 * delta2
