import numpy as np


def fold_scales(X: np.ndarray, W: np.ndarray, scales: np.ndarray):
    raise NotImplementedError


def quantize_per_tensor(W: np.ndarray, n_bits: int = 4):
    raise NotImplementedError
