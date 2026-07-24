import numpy as np

def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(z, dtype=np.float64)

    # Positive branch: use exp(-z), which is safe for large positive z
    if np.any(pos_mask):
        exp_neg_z = np.exp(-z[pos_mask])
        out[pos_mask] = 1.0 / (1.0 + exp_neg_z)

    # Negative branch: use exp(z), which is safe for large negative z
    if np.any(neg_mask):
        exp_z = np.exp(z[neg_mask])
        out[neg_mask] = exp_z / (1.0 + exp_z)

    return out
