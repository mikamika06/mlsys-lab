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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_prep": 0.0, "catches_broken_server": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import lora_pipe.engine as eng

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

    good_prep = eng.prepare_data
    eng.prepare_data = lambda raw: []
    try:
        out["catches_broken_prep"] = 0.0 if _survives(path) else 1.0
    finally:
        eng.prepare_data = good_prep

    good_server = eng.LoraServer.handle_request
    eng.LoraServer.handle_request = lambda self, p: ""
    try:
        out["catches_broken_server"] = 0.0 if _survives(path) else 1.0
    finally:
        eng.LoraServer.handle_request = good_server

    out["faults_caught"] = out["catches_broken_prep"] + out["catches_broken_server"]
    return out
