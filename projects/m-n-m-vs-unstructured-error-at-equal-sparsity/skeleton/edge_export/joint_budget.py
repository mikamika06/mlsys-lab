import numpy as np


def palettize_weights(weights: np.ndarray, num_bits: int) -> np.ndarray:
    """Quantizes weights into 2^num_bits uniform centroids."""
    raise NotImplementedError


def evaluate_joint_error(weights: np.ndarray, n: int, m: int, num_bits: int, use_nm: bool) -> float:
    """Evaluates MSE after pruning and then palettizing remaining non-zero weights."""
    raise NotImplementedError


def find_optimal_joint_budget(weights: np.ndarray, max_effective_bits: float, bit_options: list) -> dict:
    """Finds the lowest MSE pruning and palettization strategy given an effective bit budget."""
    raise NotImplementedError
