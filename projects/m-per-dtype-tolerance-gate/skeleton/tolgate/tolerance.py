import numpy as np


def compute_reduction_rtol(dtype_str: str, k_terms: int) -> float:
    """Derive the expected relative tolerance for K-term reduction."""
    raise NotImplementedError


def evaluate_gate(actual: np.ndarray, reference: np.ndarray, dtype_str: str, k_terms: int) -> dict:
    """Evaluate whether actual tensor satisfies tolerance gates against reference."""
    raise NotImplementedError
