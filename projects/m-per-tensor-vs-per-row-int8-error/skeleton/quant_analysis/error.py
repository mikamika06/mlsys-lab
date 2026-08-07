import numpy as np


def quantize_per_tensor_int8(w: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def quantize_per_row_int8(w: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def compute_mse(w_orig: np.ndarray, w_quant: np.ndarray) -> float:
    raise NotImplementedError


def compare_error_metrics(w: np.ndarray) -> dict[str, float]:
    raise NotImplementedError
