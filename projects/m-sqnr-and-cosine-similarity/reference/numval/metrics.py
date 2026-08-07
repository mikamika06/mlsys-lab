import numpy as np


def sqnr(y_ref, y_test, eps=1e-12):
    """Computes Signal-to-Quantization-Noise Ratio in dB."""
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    signal_power = np.sum(y_ref ** 2)
    noise_power = np.sum((y_ref - y_test) ** 2)
    if noise_power < eps:
        return 120.0
    val = signal_power / noise_power
    if val <= 0:
        return -120.0
    return float(10.0 * np.log10(val))


def cosine_similarity(y_ref, y_test, eps=1e-12):
    """Computes cosine similarity between two flattened tensors."""
    y_ref = np.asarray(y_ref, dtype=np.float64).flatten()
    y_test = np.asarray(y_test, dtype=np.float64).flatten()
    dot = np.dot(y_ref, y_test)
    norm_ref = np.linalg.norm(y_ref)
    norm_test = np.linalg.norm(y_test)
    denom = norm_ref * norm_test
    if denom < eps:
        return 0.0
    return float(dot / denom)
