import numpy as np


def compute_size_ratio(original_bytes: int, quantized_bytes: int) -> float:
    return float(quantized_bytes) / float(original_bytes)


def compute_quality_delta(original_output: np.ndarray, quantized_output: np.ndarray) -> float:
    return float(np.max(np.abs(original_output - quantized_output)))
