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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import exporter.replacements as rep

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_catalog = rep.catalog_add
    def broken_catalog(op_name, fn):
        return False
    rep.catalog_add = broken_catalog
    try:
        out["faults_caught"] = 0.0 if _survives(path) else 1.0
    finally:
        rep.catalog_add = good_catalog
    return out
