import numpy as np

def softmax_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits / T
    shift = np.max(z)
    exp_z = np.exp(z - shift)
    return exp_z / np.sum(exp_z)
