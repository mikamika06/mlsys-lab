import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_unimplemented": 0.0, "catches_fallback": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import mps.engine as eng_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_list = eng_mod.Engine.list_unimplemented_ops

    def broken_list(self, graph):
        return []

    eng_mod.Engine.list_unimplemented_ops = broken_list
    try:
        out["catches_unimplemented"] = 0.0 if _survives(path) else 1.0
    finally:
        eng_mod.Engine.list_unimplemented_ops = good_list

    good_run = eng_mod.Engine.run

    def broken_run(self, graph):
        return [{"name": "test", "duration": 1.0, "fallback": False}]

    eng_mod.Engine.run = broken_run
    try:
        out["catches_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        eng_mod.Engine.run = good_run

    out["faults_caught"] = out["catches_unimplemented"] + out["catches_fallback"]
    return out
