import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_verify": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        return out

    out["has_tests"] = 1.0
    try:
        if not _run(path):
            return out
    except Exception:
        return out
    out["passes_on_good"] = 1.0

    import exporter.optimizer as opt
    orig = opt.verify_output
    opt.verify_output = lambda t, o: 1.0

    try:
        out["catches_broken_verify"] = 0.0 if _survives(path) else 1.0
    finally:
        opt.verify_output = orig

    out["faults_caught"] = out["catches_broken_verify"]
    return out
