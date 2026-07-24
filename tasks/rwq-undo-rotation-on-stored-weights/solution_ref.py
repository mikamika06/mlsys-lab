import numpy as np

def reconstruct_weights(W_quantized, H, scale, zero_point):
    W_dequant = (W_quantized.astype(np.float64) - zero_point) * scale
    return H.T @ W_dequant
