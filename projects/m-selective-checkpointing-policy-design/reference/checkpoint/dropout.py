import numpy as np

def verify_dropout_consistency(mask_fwd, mask_bwd):
    return np.array_equal(mask_fwd, mask_bwd)
