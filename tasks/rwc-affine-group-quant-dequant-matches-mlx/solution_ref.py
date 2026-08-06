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
    n_rows = weights.shape[0]
    n_cols = weights.shape[1]
    n_groups = (n_rows + group_size - 1) // group_size

    q_codes = np.empty_like(weights, dtype=np.int8)
    recon = np.empty_like(weights, dtype=np.float64)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_rows)

        min_val = 0.0
        max_val = 0.0
        first = True
        for r in range(start, end):
            for c in range(n_cols):
                val = weights[r, c]
                if first:
                    min_val = val
                    max_val = val
                    first = False
                else:
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val

        if max_val != min_val:
            scale = (max_val - min_val) / 255.0
        else:
            scale = 1.0

        for r in range(start, end):
            for c in range(n_cols):
                q_val = (weights[r, c] - min_val) / scale
                rounded = round(q_val)
                if rounded < -128:
                    q_c = -128
                elif rounded > 127:
                    q_c = 127
                else:
                    q_c = int(rounded)

                q_codes[r, c] = q_c
                recon[r, c] = scale * q_c + min_val

    return q_codes, recon
