import numpy as np

def diagnose_rope(config, inv_freq):
    dim = config["head_dim"]
    theta = config["rope_theta"]
    expected = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    err = np.max(np.abs(expected - np.array(inv_freq)) / (expected + 1e-9))
    return bool(err > 1e-4)
