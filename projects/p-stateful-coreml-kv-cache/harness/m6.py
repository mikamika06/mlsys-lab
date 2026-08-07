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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        return out

    import edge_model.runtime as rt

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out
    if first is None:
        out["has_tests"] = 1.0
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_step = rt.StatefulRunner.step

    def broken_step(self, token):
        res, l = good_step(self, token)
        return res, l + 999

    rt.StatefulRunner.step = broken_step
    try:
        caught = 0 if _survives(path) else 1
        out["faults_caught"] = float(caught)
    finally:
        rt.StatefulRunner.step = good_step

    return out
