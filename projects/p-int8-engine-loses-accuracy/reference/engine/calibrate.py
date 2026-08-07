import numpy as np

def calibrate_scales(dataset):
    flattened = [np.max(np.abs(d)) for d in dataset]
    max_val = float(np.max(flattened)) if flattened else 1.0
    return max_val / 127.0
