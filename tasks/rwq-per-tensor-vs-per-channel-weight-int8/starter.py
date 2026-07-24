import numpy as np


def int8_mse_per_tensor_vs_per_channel(W: np.ndarray):
    """Compare symmetric int8 reconstruction MSE: one tensor-wide scale
    vs one scale per output row (per channel).

    W: (out_features, in_features) float64 weight matrix.

    Per-tensor: scale = max(|W|) / 127 over the whole matrix, reconstruct,
    compute MSE over all elements.
    Per-channel: for each row i, scale_i = max(|W[i,:]|) / 127, reconstruct
    that row, compute MSE over all elements (all rows combined).

    Symmetric int8 quantizer for a group g:
        scale = max(|g|) / 127
        codes = clip(round(g / scale), -127, 127)
        g_hat = codes * scale

    Returns (mse_per_tensor, mse_per_channel).
    """
    raise NotImplementedError('your code here')
