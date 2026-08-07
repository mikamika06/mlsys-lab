import importlib.util
import os
import ref

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_unsupported": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        return out

    import edge.partitioner as ep

    try:
        first = _run(path)
    except Exception:
        return out
    if first is None:
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_find = ep.find_unsupported_ops
    ep.find_unsupported_ops = lambda p: []
    try:
        out["catches_broken_unsupported"] = 0.0 if _survives(path) else 1.0
    finally:
        ep.find_unsupported_ops = good_find

    out["faults_caught"] = out["catches_broken_unsupported"]
    return out
