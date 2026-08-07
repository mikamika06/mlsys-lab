import numpy as np
from quant.hessian import compute_hessian


def find_convergence_sample_count(loader, oracle_h: np.ndarray, rel_tol: float) -> int:
  accumulated = []
  count = 0
  for batch in loader:
    accumulated.append(batch)
    count += len(batch)
    concat = np.concatenate(accumulated, axis=0)
    current_h = compute_hessian(concat)
    err = np.linalg.norm(current_h - oracle_h, ord="fro") / np.linalg.norm(oracle_h, ord="fro")
    if err <= rel_tol:
      return count
  return count
