import importlib.util
import os


def _run(path):
  spec = importlib.util.spec_from_file_location("learner_regression", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
  if not fns:
    return None
  for fn in fns:
    fn()
  return True


def _survives(path):
  try:
    return _run(path) is True
  except Exception:
    return False


def check(workdir):
  out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_affine": 0.0}
  path = os.path.join(workdir, "tests", "test_regression.py")
  if not os.path.isfile(path):
    out["_note"] = "tests/test_regression.py is missing"
    return out

  try:
    first = _run(path)
  except Exception as e:
    out["has_tests"] = 1.0
    out["_note"] = f"the tests fail on a correct reference: {type(e).__name__}: {str(e)[:120]}"
    return out

  if first is None:
    out["_note"] = "no test_* functions found"
    return out

  out["has_tests"] = 1.0
  out["passes_on_good"] = 1.0

  import qcore.derive as d

  good_fn = d.derive_affine_params

  def broken_derive_affine_params(w, num_bits=4):
    qmax = (1 << (num_bits - 1)) - 1
    max_val = float(max(abs(float(w.max())), abs(float(w.min()))))
    scale = max_val / float(qmax)
    zero_point = 0
    return float(scale), int(zero_point)

  d.derive_affine_params = broken_derive_affine_params
  import qcore.stats

  qcore.stats.derive_affine_params = broken_derive_affine_params

  try:
    out["catches_broken_affine"] = 0.0 if _survives(path) else 1.0
  finally:
    d.derive_affine_params = good_fn
    qcore.stats.derive_affine_params = good_fn

  return out
