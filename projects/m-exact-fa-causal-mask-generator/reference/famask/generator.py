import numpy as np


def generate_causal_mask(sq, sk, alignment="top-left"):
    q_indices = np.arange(sq)[:, None]
    k_indices = np.arange(sk)[None, :]
    if alignment == "top-left":
        mask = q_indices >= (k_indices - (sk - sq))
    elif alignment == "bottom-right":
        mask = q_indices + (sk - sq) >= k_indices
    else:
        raise ValueError(f"Unknown alignment: {alignment}")
    return mask
