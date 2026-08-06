import numpy as np


def dora_extra_parameters(d_in: int, d_out: int, r: int) -> int:
    """Return extra parameters of DoRA over plain LoRA."""
    return d_out


def dora_forward(w: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, alpha: float, x: np.ndarray) -> np.ndarray:
    """Compute DoRA forward pass."""
    r_val = a.shape[0]
    scale = alpha / r_val
    delta_w = np.matmul(b, a) * scale
    w_total = w + delta_w
    norms = np.linalg.norm(w_total, axis=1, keepdims=True)
    normalized_w = w_total / (norms + 1e-12)
    decomposed_w = g[:, None] * normalized_w
    return np.matmul(x, decomposed_w.T)
