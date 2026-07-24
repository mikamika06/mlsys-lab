import numpy as np

def affine_group_quant_dequant(weights: np.ndarray, group_size: int = 64):
    """
    Quantize and dequantize weights in contiguous groups.

    Parameters
    ----------
    weights : np.ndarray
        Input weight matrix of shape (N, D).
    group_size : int, optional
        Number of rows per quantization group. Default is 64.

    Returns
    -------
    q_codes : np.ndarray
        Integer codes of dtype np.int8.
    recon : np.ndarray
        Dequantized weights of dtype float64.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_groups = (weights.shape[0] + group_size - 1) // group_size

    q_codes = np.empty_like(weights, dtype=np.int8)
    recon = np.empty_like(weights, dtype=np.float64)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, weights.shape[0])
        group = weights[start:end]

        bias = group.min()
        scale = (group.max() - bias) / 255.0 if group.max() != bias else 1.0

        q = np.round((group - bias) / scale)
        q_clipped = np.clip(q, -128, 127).astype(np.int8)

        recon_group = scale * q_clipped + bias

        q_codes[start:end] = q_clipped
        recon[start:end] = recon_group

    return q_codes, recon
