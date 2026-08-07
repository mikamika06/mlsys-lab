import numpy as np

def measure_loss(orig_logits: np.ndarray, quant_logits: np.ndarray) -> float:
    diff = orig_logits - quant_logits
    return float(np.mean(diff ** 2))
