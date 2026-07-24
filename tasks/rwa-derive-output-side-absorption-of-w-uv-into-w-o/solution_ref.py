import numpy as np


def absorb_w_uv(W_O, W_UV, P, c_V):
    W_O = np.asarray(W_O, dtype=np.float64)
    W_UV = np.asarray(W_UV, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    c_V = np.asarray(c_V, dtype=np.float64)

    absorbed_output = W_O @ W_UV
    latent_attention = P @ c_V
    return latent_attention @ absorbed_output.T
