import numpy as np


def compute_quantization_error(w: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def svd_correction(error_matrix: np.ndarray, rank: int):
    raise NotImplementedError


def apply_corrected_quantization(w: np.ndarray, w_q: np.ndarray, rank: int):
    raise NotImplementedError
