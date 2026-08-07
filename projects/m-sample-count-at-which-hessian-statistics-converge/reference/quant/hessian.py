import numpy as np


def compute_hessian(activations: np.ndarray) -> np.ndarray:
  return np.dot(activations.T, activations) / activations.shape[0]


def domain_discrepancy(matched_h: np.ndarray, mismatched_h: np.ndarray) -> float:
  diff = np.linalg.norm(matched_h - mismatched_h, ord="fro")
  norm_ref = np.linalg.norm(matched_h, ord="fro")
  if norm_ref == 0:
    return float(diff)
  return float(diff / norm_ref)
