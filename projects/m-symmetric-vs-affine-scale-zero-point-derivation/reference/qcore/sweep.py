import numpy as np
from qcore.stats import quantize_blockwise


def compute_rel_err(w: np.ndarray, w_hat: np.ndarray):
  """Compute relative L2 error norm(w - w_hat) / norm(w)."""
  num = float(np.linalg.norm(w - w_hat))
  den = float(np.linalg.norm(w))
  if den == 0.0:
    return 0.0
  return float(num / den)


def run_block_size_sweep(w: np.ndarray, block_sizes=None, mode: str = "affine"):
  """Run error sweep across block sizes."""
  if block_sizes is None:
    block_sizes = [16, 32, 64, 128]
  errs = {}
  for bs in block_sizes:
    w_hat = quantize_blockwise(w, block_size=bs, mode=mode)
    errs[bs] = compute_rel_err(w, w_hat)
  return errs
