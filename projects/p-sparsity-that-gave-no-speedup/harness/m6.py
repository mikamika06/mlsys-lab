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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_false_negative": 0.0, "catches_no_compression": 0.0,
           "faults_caught": 0.0}

    if not os.path.isfile(path):
        out["_note"] = "missing test_regression.py"
        return out

    import sparsity.core as core
    import sparsity.checkpoint as chk

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_is_2_4 = core.is_2_4
    core.is_2_4 = lambda m: False
    try:
        out["catches_false_negative"] = 0.0 if _survives(path) else 1.0
    finally:
        core.is_2_4 = good_is_2_4

    good_chk = chk.checkpoint_size
    chk.checkpoint_size = lambda m: float(m.size * 2)
    try:
        out["catches_no_compression"] = 0.0 if _survives(path) else 1.0
    finally:
        chk.checkpoint_size = good_chk

    out["faults_caught"] = out["catches_false_negative"] + out["catches_no_compression"]
    return out
