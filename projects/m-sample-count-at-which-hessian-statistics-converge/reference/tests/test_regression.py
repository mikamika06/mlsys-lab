import numpy as np
from quant.convergence import find_convergence_sample_count
from quant.hessian import compute_hessian
from quant.loader import DeterministicLoader


def test_convergence_threshold():
  rng = np.random.default_rng(42)
  data = rng.normal(size=(200, 16))
  oracle_h = compute_hessian(data)
  loader = DeterministicLoader(data, seed=42, batch_size=20)
  threshold = find_convergence_sample_count(loader, oracle_h, rel_tol=1e-5)
  assert threshold > 0
  assert threshold <= 200


def test_loader_repeatability():
  rng = np.random.default_rng(123)
  data = rng.normal(size=(100, 8))
  l1 = DeterministicLoader(data, seed=99, batch_size=10)
  l2 = DeterministicLoader(data, seed=99, batch_size=10)
  b1 = list(l1)
  b2 = list(l2)
  for x, y in zip(b1, b2):
    np.testing.assert_array_equal(x, y)
