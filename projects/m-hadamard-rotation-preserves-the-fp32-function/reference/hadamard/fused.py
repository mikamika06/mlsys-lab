import numpy as np

def rms_norm_weight_fuse(weight, h_matrix):
    return np.matmul(h_matrix.T, weight)
