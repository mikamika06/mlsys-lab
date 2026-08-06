import torch


def compute_lora_gradients(X, W, A, B, alpha, dL_dY):
    r = A.shape[0]
    scaling = alpha / r

    # Forward pass recomputation
    # X: [batch, in_features]
    # W: [out_features, in_features]
    # A: [r, in_features]
    # B: [out_features, r]
    # dL_dY: [batch, out_features]

    # Y = X @ W.T + (X @ A.T @ B.T) * scaling
    # dL/dB = dL/dY.T @ (X @ A.T) * scaling -> shape [out_features, r]
    # dL/dA = (dL/dY @ B).T @ X * scaling -> shape [r, in_features]

    xA_T = X @ A.T
    dL_dB = (dL_dY.T @ xA_T) * scaling

    dL_dB_scaled = dL_dY * scaling
    dL_dA = (dL_dB_scaled @ B).T @ X

    return dL_dA, dL_dB
