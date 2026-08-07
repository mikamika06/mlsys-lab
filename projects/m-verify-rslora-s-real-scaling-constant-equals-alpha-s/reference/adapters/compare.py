import numpy as np

def compare_magnitudes(x: np.ndarray, w_a: np.ndarray, w_b: np.ndarray, alpha: float, rank: int):
    lora_out = np.dot(np.dot(x, w_a.T), w_b.T) * (alpha / rank)
    rslora_out = np.dot(np.dot(x, w_a.T), w_b.T) * (alpha / np.sqrt(rank))
    return lora_out, rslora_out

def compare_parameters(in_features: int, out_features: int, rank: int):
    lora_params = rank * in_features + out_features * rank
    ia3_params = out_features
    return lora_params, ia3_params
