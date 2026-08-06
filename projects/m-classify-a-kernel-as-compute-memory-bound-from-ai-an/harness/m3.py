import importlib.util
import os
import ref

def run(path):
spec = importlib.util.spec_from_file_location("learner_regression", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fns = [getattr(mod, n) for n in dir(mod)
if n.startswith("test") and callable(getattr(mod, n))]
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
out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_inverted_ranking": 0.0}
path = os.path.join(workdir, "tests", "test_regression.py")
if not os.path.isfile(path):
out["_note"] = "tests/test_regression.py is missing"
return out
try:
first = _run(path)
except Exception as e:
out["has_tests"] = 1.0
out["_note"] = f"tests fail on correct implementation: {type(e).name}: {str(e)[:120]}"
return out
if first is None:
out["note"] = "no test* functions found"
return out
out["has_tests"] = 1.0
out["passes_on_good"] = 1.0

import roofline.rank as r
good = r.rank_kernels

def inverted_rank(kernels):
    res = good(kernels)
    return list(reversed(res))

r.rank_kernels = inverted_rank
import roofline
roofline.rank_kernels = inverted_rank
try:
    out["catches_inverted_ranking"] = 0.0 if _survives(path) else 1.0
finally:
    r.rank_kernels = good
    roofline.rank_kernels = good
return out
