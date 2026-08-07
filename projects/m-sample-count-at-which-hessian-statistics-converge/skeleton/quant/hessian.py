import numpy as np


def compute_hessian(activations: np.ndarray) -> np.ndarray:
  raise NotImplementedError


def domain_discrepancy(matched_h: np.ndarray, mismatched_h: np.ndarray) -> float:
  raise NotImplementedError
