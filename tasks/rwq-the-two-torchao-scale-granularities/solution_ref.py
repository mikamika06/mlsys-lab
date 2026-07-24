import numpy as np

def torchao_scale_granularities(W, X):
    weight_scales = np.linalg.norm(W, axis=1)
    activation_scales = np.linalg.norm(X, axis=1)
    return weight_scales, activation_scales
