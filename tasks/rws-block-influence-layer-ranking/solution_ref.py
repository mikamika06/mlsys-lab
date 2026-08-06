import numpy as np
import math

def block_influence_ranking(
    x_in: np.ndarray,
    x_out: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """
    Compute the block influence scores and ranking for each layer.

    Parameters
    ----------
    x_in : np.ndarray
        Input activations of shape (batch, layers, features).
    x_out : np.ndarray
        Output activations of shape (batch, layers, features).

    Returns
    -------
    influences : np.ndarray
        1‑D array of length `layers` with influence scores (dtype float64).
    ranking : list[int]
        Layer indices sorted in descending order of influence.
    """
    batch = x_in.shape[0]
    layers = x_in.shape[1]
    features = x_in.shape[2]
    eps = 1e-12

    influences_list = []
    for l in range(layers):
        sum_cos = 0.0
        for b in range(batch):
            sum_sq_in = 0.0
            sum_sq_out = 0.0
            dot_val = 0.0
            for f in range(features):
                val_in = x_in[b, l, f]
                val_out = x_out[b, l, f]
                sum_sq_in += val_in * val_in
                sum_sq_out += val_out * val_out
                dot_val += val_in * val_out
            
            norm_in = math.sqrt(sum_sq_in) + eps
            norm_out = math.sqrt(sum_sq_out) + eps
            cos_val = dot_val / (norm_in * norm_out)
            sum_cos += cos_val
        
        mean_cos = sum_cos / batch
        influences_list.append(1.0 - mean_cos)

    influences = np.array(influences_list, dtype=np.float64)
    ranking = sorted(range(layers), key=lambda i: influences[i], reverse=True)

    return influences, ranking
