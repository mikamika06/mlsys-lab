import numpy as np

def verify_dual_entry(cfg, inputs):
    return np.sum(inputs * 2.5, axis=-1)
