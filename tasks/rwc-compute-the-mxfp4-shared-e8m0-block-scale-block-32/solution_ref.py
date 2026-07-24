import numpy as np

def compute_shared_e8m0_scale(weights):
    amax = np.max(np.abs(weights), axis=1)
    exponents = np.maximum(0, np.ceil(np.log2(amax / 6.0))).astype(np.int32)
    return exponents
