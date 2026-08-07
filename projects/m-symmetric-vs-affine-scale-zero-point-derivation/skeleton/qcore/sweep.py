import numpy as np


def compute_rel_err(w: np.ndarray, w_hat: np.ndarray):
  """Compute relative L2 error norm(w - w_hat) / norm(w)."""
  raise NotImplementedError


def run_block_size_sweep(w: np.ndarray, block_sizes=None, mode: str = "affine"):
  """Run error sweep across block sizes."""
  raise NotImplementedError
