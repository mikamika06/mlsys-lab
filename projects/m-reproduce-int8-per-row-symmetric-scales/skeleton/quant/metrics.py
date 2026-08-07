import numpy as np


def compute_size_ratio(original_bytes: int, quantized_bytes: int) -> float:
    raise NotImplementedError


def compute_quality_delta(original_output: np.ndarray, quantized_output: np.ndarray) -> float:
    raise NotImplementedError
