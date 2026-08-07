import numpy as np


def apply_unstructured_pruning(weights: np.ndarray, sparsity_ratio: float) -> np.ndarray:
    """Applies unstructured magnitude pruning to target sparsity ratio."""
    raise NotImplementedError


def apply_nm_pruning(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    """Applies n:m block pruning along the last dimension of weights."""
    raise NotImplementedError


def compare_sparsity_error(weights: np.ndarray, n: int, m: int) -> dict:
    """Computes MSE reconstruction error for n:m vs unstructured pruning at equal sparsity."""
    raise NotImplementedError
