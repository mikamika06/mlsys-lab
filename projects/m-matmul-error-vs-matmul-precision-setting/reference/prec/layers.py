import numpy as np
from prec.analysis import compute_matmul_error

def evaluate_layer_bf16_safety(layer_weights: np.ndarray, threshold: float) -> bool:
    x = np.linspace(-1.0, 1.0, layer_weights.shape[0], dtype=np.float32)
    err = compute_matmul_error(layer_weights, x[:, None], "bf16")
    return bool(err <= threshold)
