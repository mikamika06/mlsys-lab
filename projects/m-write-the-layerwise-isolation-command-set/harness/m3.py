import importlib.util
import os
import sys


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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_tolerance": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import polyiso.divergence as div_mod

    orig_find = div_mod.find_first_divergent_layer

    def broken_find_first_divergent_layer(layer_outputs, rtol=1e-3, atol=1e-5):
        return orig_find(layer_outputs, rtol=0.0, atol=0.0)

    div_mod.find_first_divergent_layer = broken_find_first_divergent_layer

    try:
        out["catches_invalid_tolerance"] = 0.0 if _survives(path) else 1.0
    finally:
        div_mod.find_first_divergent_layer = orig_find

    return out
