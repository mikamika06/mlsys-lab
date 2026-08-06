import numpy as np


def compute_scaling_factor(alpha, r, mode="lora"):
    if mode == "lora":
        return alpha / r
    elif mode == "naive":
        return alpha
    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_lora_scaling(x, weight_a, weight_b, alpha, r, mode="lora"):
    scale = compute_scaling_factor(alpha, r, mode=mode)
    h = np.dot(x, weight_a.T)
    delta = np.dot(h, weight_b.T)
    return scale * delta
